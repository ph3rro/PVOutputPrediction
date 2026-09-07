"""Fast decoder for the UTC timestamp burned into UOH video frames."""

from __future__ import annotations

import datetime as dt
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu

THRESHOLD = 170
GLYPH_HEIGHT = 8
DATE_TO_TIME_Y = 13
_DIGIT_CELL_WIDTH = 6
_N_DIGITS_IN_TIME = 6
_PACK_POWERS = np.left_shift(
    np.uint64(1), np.arange(48, dtype=np.uint64))
_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})$")


def _positions(text: str, x0: int) -> list[int]:
    """Character positions for the proportional UOH bitmap font."""
    positions = []
    x = x0
    for char in text:
        positions.append(x)
        x += 3 if char in "/:" else 6
    return positions


def _pack_cells(cells: np.ndarray) -> np.ndarray:
    flat = cells.reshape(cells.shape[:-2] + (48,)).astype(np.uint64)
    return np.sum(flat * _PACK_POWERS, axis=-1, dtype=np.uint64)


def _date_from_stem(stem: str) -> dt.date:
    match = _DATE_RE.search(stem)
    if match is None:
        raise ValueError(f"Cannot parse date from UOH video stem: {stem}")
    return dt.date(*(int(value) for value in match.groups()))


@dataclass(frozen=True)
class MinuteFrame:
    frame_index: int
    target_timestamp: int
    actual_timestamp: int
    offset_seconds: int
    ocr_score: int


class UOHTimestampDecoder:
    """Decode the fixed UOH OSD using glyphs learned from known video dates."""

    def __init__(self, reference_two: np.ndarray,
                 templates: dict[str, tuple[int, ...]]):
        self.reference_two = reference_two.astype(bool)
        self.templates = templates

        owners: dict[int, set[str]] = {}
        for digit, codes in templates.items():
            for code in codes:
                owners.setdefault(code, set()).add(digit)
        self.exact = {
            code: next(iter(digits))
            for code, digits in owners.items() if len(digits) == 1
        }
        self.flat_templates = [
            (code, digit)
            for digit, codes in templates.items()
            for code in codes
        ]

    @classmethod
    def bootstrap(cls, video_paths: list[Path],
                  max_videos: int = 256) -> "UOHTimestampDecoder":
        """Learn compressed glyph variants from date strings in first frames."""
        if not video_paths:
            raise ValueError("At least one UOH video is required")

        first = VideoReader(
            str(video_paths[0]), ctx=cpu(0), num_threads=1)[0].asnumpy()
        # All UOH overlays begin near (4, 6) with the year-leading digit 2.
        reference_two = (
            first[6:6 + GLYPH_HEIGHT, 4:4 + _DIGIT_CELL_WIDTH, 1]
            > THRESHOLD
        )
        provisional = cls(reference_two, {str(i): () for i in range(10)})
        learned: dict[str, set[int]] = {str(i): set() for i in range(10)}

        count = min(max_videos, len(video_paths))
        sample_indices = np.linspace(
            0, len(video_paths) - 1, count, dtype=int)
        for index in sample_indices:
            path = video_paths[int(index)]
            frame = VideoReader(
                str(path), ctx=cpu(0), num_threads=1)[0].asnumpy()
            x0, y0, _ = provisional.find_origin(frame)
            mask = frame[
                y0:y0 + GLYPH_HEIGHT, :, 1] > THRESHOLD
            date_text = _date_from_stem(path.stem).strftime("%Y/%m/%d")
            for x, char in zip(_positions(date_text, x0), date_text):
                if char.isdigit():
                    code = int(_pack_cells(
                        mask[:, x:x + _DIGIT_CELL_WIDTH][None])[0])
                    learned[char].add(code)

        missing = [digit for digit, codes in learned.items() if not codes]
        if missing:
            raise RuntimeError(
                f"Could not learn UOH timestamp digits: {missing}")
        return cls(reference_two, {
            digit: tuple(sorted(codes)) for digit, codes in learned.items()
        })

    def find_origin(self, frame: np.ndarray) -> tuple[int, int, int]:
        """Locate the date-line origin by matching its leading digit 2."""
        mask = frame[:, :, 1] > THRESHOLD
        score, x0, y0 = min(
            (
                int(np.count_nonzero(
                    mask[y:y + GLYPH_HEIGHT,
                         x:x + _DIGIT_CELL_WIDTH] != self.reference_two)),
                x,
                y,
            )
            for y in range(3, 10)
            for x in range(1, 8)
        )
        return x0, y0, score

    def _classify(self, code: int) -> tuple[str, int]:
        exact = self.exact.get(code)
        if exact is not None:
            return exact, 0
        distance, digit = min(
            ((code ^ template).bit_count(), label)
            for template, label in self.flat_templates
        )
        return digit, distance

    def _decode_codes(self, codes: np.ndarray) -> tuple[list[str], list[int]]:
        texts = []
        scores = []
        for row in codes:
            digits = []
            score = 0
            for code in row:
                digit, distance = self._classify(int(code))
                digits.append(digit)
                score += distance
            texts.append(
                f"{digits[0]}{digits[1]}:"
                f"{digits[2]}{digits[3]}:"
                f"{digits[4]}{digits[5]}")
            scores.append(score)
        return texts, scores

    def decode_time_batch(self, frames: np.ndarray, x0: int,
                          time_y: int) -> tuple[list[str], list[int]]:
        """Decode HH:MM:SS for an RGB frame batch."""
        mask = (
            frames[:, time_y:time_y + GLYPH_HEIGHT, :, 1] > THRESHOLD)
        digit_positions = [
            x for x, char in zip(
                _positions("00:00:00", x0), "00:00:00")
            if char.isdigit()
        ]
        cells = np.stack([
            mask[:, :, x:x + _DIGIT_CELL_WIDTH]
            for x in digit_positions
        ], axis=1)
        return self._decode_codes(_pack_cells(cells))

    def select_minute_frames(
        self,
        video_path: Path,
        batch_size: int = 128,
        max_ocr_score: int = 12,
    ) -> list[MinuteFrame]:
        """Choose the frame closest to each UTC minute in a UOH video."""
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        first = vr[0].asnumpy()
        x0, date_y, origin_score = self.find_origin(first)
        if origin_score > 8:
            raise RuntimeError(
                f"Timestamp overlay alignment score is {origin_score}")

        video_date = _date_from_stem(video_path.stem)
        midnight = int(dt.datetime.combine(
            video_date, dt.time(), tzinfo=dt.timezone.utc).timestamp())
        best: dict[int, MinuteFrame] = {}

        for start in range(0, len(vr), batch_size):
            indices = list(range(start, min(start + batch_size, len(vr))))
            frames = vr.get_batch(indices).asnumpy()
            texts, scores = self.decode_time_batch(
                frames, x0=x0, time_y=date_y + DATE_TO_TIME_Y)
            for frame_index, text, score in zip(indices, texts, scores):
                if score > max_ocr_score:
                    continue
                hour, minute, second = (int(part) for part in text.split(":"))
                if hour > 23 or minute > 59 or second > 59:
                    continue
                seconds = hour * 3600 + minute * 60 + second
                target_seconds = ((seconds + 30) // 60) * 60
                offset = seconds - target_seconds
                target = midnight + target_seconds
                candidate = MinuteFrame(
                    frame_index=frame_index,
                    target_timestamp=target,
                    actual_timestamp=midnight + seconds,
                    offset_seconds=offset,
                    ocr_score=score,
                )
                current = best.get(target)
                if current is None or (
                    abs(candidate.offset_seconds),
                    candidate.ocr_score,
                    candidate.frame_index,
                ) < (
                    abs(current.offset_seconds),
                    current.ocr_score,
                    current.frame_index,
                ):
                    best[target] = candidate

        selected = sorted(best.values(), key=lambda item: item.target_timestamp)

        # Keep the longest time-ordered frame sequence. This removes isolated
        # OCR errors (for example, reading 19:27 as 10:27) without assuming
        # that the source has no genuine gaps.
        tails: list[int] = []
        tail_indices: list[int] = []
        previous = [-1] * len(selected)
        for index, item in enumerate(selected):
            position = bisect_left(tails, item.frame_index)
            if position == len(tails):
                tails.append(item.frame_index)
                tail_indices.append(index)
            else:
                tails[position] = item.frame_index
                tail_indices[position] = index
            if position:
                previous[index] = tail_indices[position - 1]

        keep = []
        index = tail_indices[-1] if tail_indices else -1
        while index >= 0:
            keep.append(selected[index])
            index = previous[index]
        return list(reversed(keep))
