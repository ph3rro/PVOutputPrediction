#!/usr/bin/env python
"""Extract every frame of the PVotSky sky-camera videos as PNGs named by the
timestamp shown in the on-screen "Rec. Time:" overlay.

The overlay is rendered with a fixed-pitch (16 px) bitmap font and the videos
are lossless (CRF 0), so each character is a pixel-exact bitmap. Instead of
fuzzy OCR, this script:

  1. Bootstraps a glyph-template dictionary automatically: for a handful of
     videos it locates the date string in the overlay (the date is known from
     the video filename) and harvests the exact bitmaps of the digits and the
     '-', ':' and '.' separators. Templates are cached to a JSON file.
  2. For every video, finds the overlay line containing a decodable
     "YYYY-MM-DD HH:MM:SS.mmm" timestamp, then decodes it in every frame by
     exact bitmap lookup, crops the frame to the sky disk (CROP), downscales
     it to OUT_SIZE x OUT_SIZE, and writes it to
     <out_root>/<year>/<month>/<date>/<YYYYMMDD_HHMMSS_mmm>.png

Run e.g.:
    python extract_pvotsky_frames.py --dates 20220801            # one day
    python extract_pvotsky_frames.py --start 20230101 --end 20230131
    python extract_pvotsky_frames.py                             # everything

The full dataset is ~570k frames; at 224x224 the PNG output is roughly
50-100 KB per frame, i.e. on the order of 30-60 GB for everything.
"""

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "data" / "raw_data" / "pangaea" / "PVotSky"
DEFAULT_OUT = SCRIPT_DIR / "data" / "raw_data" / "pangaea" / "PVotSky_frames"
TEMPLATE_CACHE = SCRIPT_DIR / "pvotsky_osd_templates.json"

PITCH = 16          # fixed-pitch font cell width in pixels
OSD_H = 240         # overlay search window (top-left corner of the frame)
OSD_W = 576         # 36 cells; stays clear of the fisheye image circle
N_CELLS = OSD_W // PITCH
THRESH = 180        # glyph pixels are >=230, background in the overlay area ~0

TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})")
TEMPLATE_CHARS = "0123456789-:."

# Square (x0, y0, x1, y1) box around the fisheye sky disk (see
# video_only_data_preprocess.ipynb) and the output resolution: each saved
# frame is cropped to this box, then downscaled to OUT_SIZE x OUT_SIZE.
CROP = (88, 38, 1986, 1936)
OUT_SIZE = 224


# --------------------------------------------------------------------------
# overlay parsing primitives
# --------------------------------------------------------------------------

def osd_mask(frame):
    """Binary mask of the overlay search window (green channel is enough)."""
    return frame[:OSD_H, :OSD_W, 1] > THRESH


def find_text_bands(mask):
    """Row ranges (r0, r1) of horizontal text lines in the mask."""
    occ = mask.sum(axis=1) > 0
    bands, start = [], None
    for y in range(len(occ) + 1):
        filled = occ[y] if y < len(occ) else False
        if filled and start is None:
            start = y
        elif not filled and start is not None:
            bands.append((start, y))
            start = None
    return bands


def band_cells(mask, r0, r1):
    """Per-cell glyph bitmaps of a text band as bytes (None for empty cells)."""
    band = mask[r0:r1].astype(np.uint8)
    cells = []
    for k in range(N_CELLS):
        cell = band[:, k * PITCH:(k + 1) * PITCH]
        cells.append(cell.tobytes() if cell.any() else None)
    return cells


def decode_cells(cells, templates):
    """Map cell bitmaps to characters (' ' = empty, '?' = unknown glyph)."""
    return "".join(
        " " if c is None else templates.get(c, "?") for c in cells
    )


def decode_frame(mask, r0, r1, templates):
    """Decode the timestamp from a known overlay line; None if not found."""
    text = decode_cells(band_cells(mask, r0, r1), templates)
    return TS_RE.search(text)


def find_timestamp_band(mask, templates):
    """Search all text bands for one containing a decodable timestamp."""
    for r0, r1 in find_text_bands(mask):
        if TS_RE.search(decode_cells(band_cells(mask, r0, r1), templates)):
            return r0, r1
    return None


# --------------------------------------------------------------------------
# template bootstrap
# --------------------------------------------------------------------------

def _slice_matches_text(cells, text, char_to_bmp, bmp_to_char):
    """Check a run of cells against a known string using glyph-equality
    constraints and any templates already learned."""
    for cell, ch in zip(cells, text):
        if cell is None:
            return False
        if ch in char_to_bmp and char_to_bmp[ch] != cell:
            return False
        if cell in bmp_to_char and bmp_to_char[cell] != ch:
            return False
    for j in range(len(text)):
        for k in range(j + 1, len(text)):
            if (text[j] == text[k]) != (cells[j] == cells[k]):
                return False
    return True


def build_templates(video_paths, cache_path):
    """Learn the bitmap of every digit and separator from the date field of
    the overlay, whose value is known from each video's filename."""
    if cache_path.exists():
        data = json.load(open(cache_path))
        return {
            bytes(bytearray(v)): ch for ch, v in data.items()
        }

    char_to_bmp, bmp_to_char = {}, {}

    def learned():
        return all(c in char_to_bmp for c in TEMPLATE_CHARS)

    for path in video_paths:
        if learned():
            break
        stem = path.stem  # e.g. 20220801
        date_txt = f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        mask = osd_mask(frame)
        for r0, r1 in find_text_bands(mask):
            cells = band_cells(mask, r0, r1)
            hits = [
                i for i in range(N_CELLS - len(date_txt) + 1)
                if _slice_matches_text(
                    cells[i:i + len(date_txt)], date_txt,
                    char_to_bmp, bmp_to_char,
                )
            ]
            if len(hits) != 1:
                continue
            i = hits[0]
            # The date is followed by " HH:MM:SS.mmm": use it to sanity-check
            # the match and to harvest the ':' and '.' separators.
            tail = cells[i + 10:i + 23]
            if len(tail) < 13 or tail[0] is not None:
                continue
            if not all(tail[j] is not None for j in range(1, 13)):
                continue
            if tail[3] != tail[6]:  # the two ':' must be identical
                continue
            pairs = list(zip(date_txt, cells[i:i + 10]))
            pairs += [(":", tail[3]), (".", tail[9])]
            for ch, bmp in pairs:
                if bmp_to_char.get(bmp, ch) != ch or \
                        char_to_bmp.get(ch, bmp) != bmp:
                    break  # inconsistent -> wrong line, learn nothing
            else:
                for ch, bmp in pairs:
                    char_to_bmp[ch] = bmp
                    bmp_to_char[bmp] = ch
                break

    if not learned():
        missing = [c for c in TEMPLATE_CHARS if c not in char_to_bmp]
        raise RuntimeError(f"template bootstrap incomplete, missing {missing}")

    json.dump(
        {ch: list(bmp) for ch, bmp in char_to_bmp.items()},
        open(cache_path, "w"),
    )
    return bmp_to_char


# --------------------------------------------------------------------------
# per-video extraction
# --------------------------------------------------------------------------

def expected_frame_count(video_path):
    sidecar = video_path.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.load(open(sidecar))["frame_count"]
        except Exception:
            pass
    return None


def process_video(args):
    video_path, out_root, templates, png_compression, overwrite = args
    video_path = Path(video_path)
    stem = video_path.stem                      # 20220801
    out_dir = Path(out_root) / stem[:4] / stem[4:6] / stem
    n_expected = expected_frame_count(video_path)

    if not overwrite and n_expected and out_dir.is_dir():
        if len(list(out_dir.glob("*.png"))) >= n_expected:
            return stem, "skipped", 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    band = None
    n_saved = n_failed = idx = 0
    imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = osd_mask(frame)
        m = decode_frame(mask, *band, templates) if band else None
        if m is None:
            band = find_timestamp_band(mask, templates)
            m = decode_frame(mask, *band, templates) if band else None
        if m is None:
            name = f"{stem}_frame{idx:05d}_UNPARSED.png"
            n_failed += 1
        else:
            y, mo, d, h, mi, s, ms = m.groups()
            name = f"{y}{mo}{d}_{h}{mi}{s}_{ms}.png"
        x0, y0, x1, y1 = CROP
        out = cv2.resize(frame[y0:y1, x0:x1], (OUT_SIZE, OUT_SIZE),
                         interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / name), out, imwrite_params)
        n_saved += 1
        idx += 1
    cap.release()

    status = "ok"
    if n_failed:
        status = f"{n_failed} unparsed"
    elif n_expected and n_saved != n_expected:
        status = f"expected {n_expected} frames, got {n_saved}"
    return stem, status, n_saved, n_failed


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="PVotSky directory containing <year>/<month>/*.mp4")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output directory for the PNG frames")
    ap.add_argument("--dates", default=None,
                    help="comma-separated list of dates (YYYYMMDD) to process")
    ap.add_argument("--start", default=None, help="first date, YYYYMMDD")
    ap.add_argument("--end", default=None, help="last date, YYYYMMDD")
    ap.add_argument("--workers", type=int, default=4,
                    help="videos processed in parallel")
    ap.add_argument("--png-compression", type=int, default=3,
                    help="PNG compression level 0-9 (higher = smaller/slower)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract videos whose output already exists")
    args = ap.parse_args()

    videos = sorted(args.root.glob("[0-9]*/[0-9]*/*.mp4"))
    if args.dates:
        wanted = set(args.dates.split(","))
        videos = [v for v in videos if v.stem in wanted]
    if args.start:
        videos = [v for v in videos if v.stem >= args.start]
    if args.end:
        videos = [v for v in videos if v.stem <= args.end]
    if not videos:
        sys.exit("no videos matched")

    print(f"{len(videos)} videos: {videos[0].stem} .. {videos[-1].stem}")
    print(f"output -> {args.out}")

    # bootstrap templates from the whole dataset so every digit is seen
    all_videos = sorted(args.root.glob("[0-9]*/[0-9]*/*.mp4"))
    templates = build_templates(all_videos, TEMPLATE_CACHE)

    tasks = [
        (str(v), str(args.out), templates, args.png_compression,
         args.overwrite)
        for v in videos
    ]
    totals = {"saved": 0, "failed": 0}
    with Pool(args.workers) as pool:
        for stem, status, n_saved, n_failed in tqdm(
            pool.imap_unordered(process_video, tasks),
            total=len(tasks), unit="video",
        ):
            totals["saved"] += n_saved
            totals["failed"] += n_failed
            if status not in ("ok", "skipped"):
                tqdm.write(f"{stem}: {status}")

    print(f"done: {totals['saved']} frames written, "
          f"{totals['failed']} with unparseable timestamps")


if __name__ == "__main__":
    main()
