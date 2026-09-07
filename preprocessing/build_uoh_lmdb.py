#!/usr/bin/env python
"""Build an LMDB of preprocessed UOH all-sky frames.

For each MP4 under the UOH raw-data directory:
  1. Optionally decode the burned-in UTC timestamp and retain the frame
     closest to each minute.
  2. Crop horizontally to [x1, x2)
  3. Pad vertically with black so the frame is square
  4. Downscale to OUT_SIZE x OUT_SIZE (area-averaged)
  5. Encode as PNG (or JPEG) and store in LMDB

LMDB layout
-----------
  Standard layout:
    key   = video stem, e.g. b'camera7_2016-05-04'
    value = pickle.dumps(list_of_encoded_frame_bytes)
  Minute-aligned layout:
    key   = b'<video stem>@<target UTC timestamp>'
    value = encoded frame bytes
    b'__timestamps__' -> pickle.dumps({stem: [target_timestamp, ...]})
  b'__keys__'   -> pickle.dumps([stem, ...])   # sorted video stems
  b'__videos__' -> pickle.dumps({stem: n_frames, ...})
  b'__meta__'   -> json metadata (crop, size, encoding, ...)

Example
-------
  python build_uoh_lmdb.py
  python build_uoh_lmdb.py --format png --workers 8
  python build_uoh_lmdb.py --format jpg --jpeg-quality 75   # much smaller
  python build_uoh_lmdb.py --minute-aligned --format jpg
  python build_uoh_lmdb.py --limit 3                        # smoke test
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import lmdb
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm

from uoh_timestamp import UOHTimestampDecoder

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = SCRIPT_DIR / "data" / "raw_data" / "uoh"
DEFAULT_OUT = SCRIPT_DIR / "data" / "uoh_lmdb_224"

X1 = 24
X2 = 630
OUT_SIZE = 224


def preprocess_frame(frame: np.ndarray,
                     x1: int = X1,
                     x2: int = X2,
                     size: int = OUT_SIZE) -> np.ndarray:
    """Crop horizontally, pad vertically to square, resize to size x size.

    Args:
        frame: HxWx3 RGB uint8.
    Returns:
        size x size x 3 RGB uint8.
    """
    cropped = frame[:, x1:x2]
    h, w = cropped.shape[:2]
    side = max(h, w)
    pad_top = (side - h) // 2
    pad_bot = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    padded = cv2.copyMakeBorder(
        cropped, pad_top, pad_bot, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    return cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)


def encode_frame(rgb: np.ndarray, fmt: str, jpeg_quality: int) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if fmt == "png":
        ok, buf = cv2.imencode(
            ".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    elif fmt == "jpg":
        ok, buf = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def minute_frame_key(stem: str, timestamp: int) -> bytes:
    return f"{stem}@{timestamp}".encode("utf-8")


def process_video(args):
    """Decode one video and return selected, encoded frames plus timestamps."""
    video_path, x1, x2, size, fmt, jpeg_quality, decoder = args
    stem = Path(video_path).stem
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        n = len(vr)
        if decoder is not None:
            selected = decoder.select_minute_frames(Path(video_path))
            frame_ids = [item.frame_index for item in selected]
            target_timestamps = [
                item.target_timestamp for item in selected]
            actual_timestamps = [
                item.actual_timestamp for item in selected]
            offsets = [item.offset_seconds for item in selected]
        else:
            frame_ids = list(range(n))
            target_timestamps = []
            actual_timestamps = []
            offsets = []

        # Batch-read for throughput; keep batches modest to bound RAM.
        batch = 64
        encoded: list[bytes] = []
        for start in range(0, len(frame_ids), batch):
            ids = frame_ids[start:start + batch]
            frames = vr.get_batch(ids).asnumpy()
            for fr in frames:
                out = preprocess_frame(fr, x1=x1, x2=x2, size=size)
                encoded.append(encode_frame(out, fmt, jpeg_quality))
        return (
            stem,
            encoded,
            target_timestamps,
            actual_timestamps,
            frame_ids,
            offsets,
            n,
            None,
        )
    except Exception as e:  # noqa: BLE001 - keep worker alive, report per-video
        return stem, [], [], [], [], [], 0, f"{type(e).__name__}: {e}"


def estimate_map_size(n_videos: int, fmt: str, jpeg_quality: int,
                      minute_aligned: bool) -> int:
    """Conservative LMDB map_size in bytes."""
    # Empirical mean bytes/frame from a midday UOH frame after preprocess.
    per_frame = {
        ("png", 0): 50_000,
        ("jpg", 70): 5_300,
        ("jpg", 75): 5_800,
        ("jpg", 80): 7_200,
        ("jpg", 85): 8_400,
        ("jpg", 90): 10_300,
    }
    bpf = per_frame.get((fmt, jpeg_quality if fmt == "jpg" else 0), 50_000)
    frames_per_video = 700 if minute_aligned else 2000
    # 1.4x headroom for LMDB/pickle overhead and unusually long videos.
    return int(n_videos * frames_per_video * bpf * 1.4) + (2 << 30)


def build_lmdb(
    video_dir: Path,
    out_dir: Path,
    x1: int,
    x2: int,
    size: int,
    fmt: str,
    jpeg_quality: int,
    workers: int,
    limit: int | None,
    map_size: int | None,
    minute_aligned: bool,
    resume: bool,
) -> None:
    all_video_paths = sorted(video_dir.glob("*.mp4"))
    if not all_video_paths:
        raise FileNotFoundError(f"No mp4s in {video_dir}")
    video_paths = all_video_paths
    if limit is not None:
        video_paths = video_paths[:limit]

    decoder = (
        UOHTimestampDecoder.bootstrap(all_video_paths)
        if minute_aligned else None
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = out_dir / "frames.lmdb"
    has_existing_data = (
        lmdb_path.exists() and any(lmdb_path.iterdir()))
    if has_existing_data and not resume:
        raise FileExistsError(
            f"{lmdb_path} already exists and is non-empty. "
            "Delete it first, choose another --out, or pass --resume."
        )
    if has_existing_data and not minute_aligned:
        raise ValueError(
            "--resume is currently supported only with --minute-aligned")
    lmdb_path.mkdir(parents=True, exist_ok=True)

    if map_size is None:
        map_size = estimate_map_size(
            len(video_paths), fmt, jpeg_quality, minute_aligned)

    print(f"Videos     : {len(video_paths)} from {video_dir}")
    print(f"Output     : {lmdb_path}")
    print(f"Preprocess : crop x=[{x1},{x2}) -> pad square -> {size}x{size}")
    print(f"Encoding   : {fmt}"
          + (f" q={jpeg_quality}" if fmt == "jpg" else ""))
    print(f"Timing     : {'closest frame to each UTC minute' if minute_aligned else 'all frames'}")
    print(f"Workers    : {workers}")
    print(f"map_size   : {map_size / (1 << 30):.1f} GiB")

    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    video_timestamps: dict[str, list[int]] = {}
    if has_existing_data:
        prefix = b"__ts__:"
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            if cursor.set_range(prefix):
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break
                    stem = key[len(prefix):].decode("utf-8")
                    video_timestamps[stem] = pickle.loads(value)
        completed = set(video_timestamps)
        video_paths = [
            path for path in video_paths if path.stem not in completed]
        print(
            f"Resume     : {len(completed)} videos complete, "
            f"{len(video_paths)} remaining")

    worker_args = [
        (str(p), x1, x2, size, fmt, jpeg_quality, decoder)
        for p in video_paths
    ]
    keys: list[str] = sorted(video_timestamps)
    video_lengths: dict[str, int] = {
        stem: len(timestamps)
        for stem, timestamps in video_timestamps.items()
    }
    errors: list[tuple[str, str]] = []
    n_frames_total = sum(video_lengths.values())
    n_source_frames_total = None if has_existing_data else 0
    max_abs_offset = None if has_existing_data else 0
    bytes_total = 0
    t0 = time.time()

    # Commit in batches so a crash mid-run still leaves usable partial data.
    commit_every = 8
    pending = []

    def flush(pending_items) -> None:
        nonlocal bytes_total
        if not pending_items:
            return
        with env.begin(write=True) as txn:
            for stem, frames, timestamps in pending_items:
                if minute_aligned:
                    for frame, timestamp in zip(frames, timestamps):
                        txn.put(minute_frame_key(stem, timestamp), frame)
                        bytes_total += len(frame)
                    txn.put(
                        f"__ts__:{stem}".encode("utf-8"),
                        pickle.dumps(
                            timestamps, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                else:
                    payload = pickle.dumps(
                        frames, protocol=pickle.HIGHEST_PROTOCOL)
                    txn.put(stem.encode("utf-8"), payload)
                    bytes_total += len(payload)
        pending_items.clear()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_video, a): a[0] for a in worker_args}
        with tqdm(total=len(futures), desc="videos", unit="vid") as pbar:
            for fut in as_completed(futures):
                (stem, frames, timestamps, actual_timestamps, frame_ids,
                 offsets, n_source, err) = fut.result()
                if err is not None:
                    errors.append((stem, err))
                    pbar.update(1)
                    continue
                keys.append(stem)
                video_lengths[stem] = len(frames)
                if minute_aligned:
                    video_timestamps[stem] = timestamps
                n_frames_total += len(frames)
                if n_source_frames_total is not None:
                    n_source_frames_total += n_source
                if offsets and max_abs_offset is not None:
                    max_abs_offset = max(
                        max_abs_offset, max(abs(value) for value in offsets))
                pending.append((stem, frames, timestamps))
                if len(pending) >= commit_every:
                    flush(pending)
                pbar.set_postfix(
                    frames=n_frames_total,
                    GiB=f"{bytes_total / (1 << 30):.1f}",
                    err=len(errors),
                )
                pbar.update(1)

    flush(pending)

    keys.sort()
    meta = {
        "source": str(video_dir.resolve()),
        "x1": x1,
        "x2": x2,
        "out_size": size,
        "format": fmt,
        "jpeg_quality": jpeg_quality if fmt == "jpg" else None,
        "n_videos": len(keys),
        "n_frames": n_frames_total,
        "n_source_frames": n_source_frames_total,
        "n_errors": len(errors),
        "layout": (
            "minute_frames_v1" if minute_aligned else "video_payload_v1"),
        "minute_aligned": minute_aligned,
        "max_minute_offset_seconds": (
            max_abs_offset if minute_aligned else None),
        "interpolation": "cv2.INTER_AREA",
        "pipeline": (
            f"crop_x[{x1}:{x2}] -> pad_vertical_square -> "
            f"resize_{size}x{size}"
        ),
    }
    with env.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
        txn.put(
            b"__videos__",
            pickle.dumps(video_lengths, protocol=pickle.HIGHEST_PROTOCOL),
        )
        if minute_aligned:
            txn.put(
                b"__timestamps__",
                pickle.dumps(
                    video_timestamps, protocol=pickle.HIGHEST_PROTOCOL),
            )
        txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))
    env.sync()
    env.close()

    # Sidecar files for convenience / VideoMAE-style listing.
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    with open(out_dir / "videos.txt", "w") as f:
        for stem in keys:
            # path-like key, start_idx, total_frames, label
            f.write(f"{stem} 0 {video_lengths[stem]} 0\n")
    if errors:
        with open(out_dir / "errors.json", "w") as f:
            json.dump(errors, f, indent=2)

    dt = time.time() - t0
    print(f"Done in {dt / 60:.1f} min")
    print(f"Wrote {len(keys)} videos / {n_frames_total} frames "
          f"({bytes_total / (1 << 30):.2f} GiB payload)")
    if errors:
        print(f"WARNING: {len(errors)} videos failed; see {out_dir / 'errors.json'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--x1", type=int, default=X1)
    ap.add_argument("--x2", type=int, default=X2)
    ap.add_argument("--size", type=int, default=OUT_SIZE)
    ap.add_argument("--format", choices=("png", "jpg"), default="png",
                    help="Frame encoding inside LMDB. PNG is lossless but "
                         "~340 GB for the full UOH set; JPG q75 is ~40 GB.")
    ap.add_argument("--jpeg-quality", type=int, default=75)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N videos (smoke test).")
    ap.add_argument("--map-size", type=int, default=None,
                    help="LMDB map_size in bytes (auto if omitted).")
    ap.add_argument(
        "--minute-aligned",
        action="store_true",
        help=(
            "Decode the burned-in UTC timestamp and keep only the frame "
            "closest to each minute."),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue a partially built minute-aligned LMDB in --out.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_lmdb(
        video_dir=args.video_dir,
        out_dir=args.out,
        x1=args.x1,
        x2=args.x2,
        size=args.size,
        fmt=args.format,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
        limit=args.limit,
        map_size=args.map_size,
        minute_aligned=args.minute_aligned,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
