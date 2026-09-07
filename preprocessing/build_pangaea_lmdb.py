#!/usr/bin/env python
"""Build an LMDB of preprocessed Pangaea / PVotSky all-sky frames.

For each MP4 under the PVotSky raw-data directory (nested year/month/):
  1. Crop to the square sky-disk ROI CROP = (x0, y0, x1, y1)
  2. Downscale to OUT_SIZE x OUT_SIZE (area-averaged)
  3. Encode as PNG (or JPEG) and store in LMDB

LMDB layout
-----------
  Standard layout:
    key   = video stem, e.g. b'20220811'
    value = pickle.dumps(list_of_encoded_frame_bytes)
  b'__len__:{stem}' -> pickle.dumps(n_frames)  # written in the same txn as the payload
  b'__keys__'   -> pickle.dumps([stem, ...])   # sorted video stems
  b'__videos__' -> pickle.dumps({stem: n_frames, ...})
  b'__meta__'   -> json metadata (crop, size, encoding, ...)

Example
-------
  python build_pangaea_lmdb.py
  python build_pangaea_lmdb.py --format png --workers 8
  python build_pangaea_lmdb.py --format jpg --jpeg-quality 75   # much smaller
  python build_pangaea_lmdb.py --limit 3                        # smoke test
  python build_pangaea_lmdb.py --format png --out ... --resume  # continue after a crash
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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = SCRIPT_DIR / "data" / "raw_data" / "pangaea" / "PVotSky"
DEFAULT_OUT = SCRIPT_DIR / "data" / "pangaea_lmdb_224"

# Square crop around the fisheye sky disk (see video_only_data_preprocess.ipynb).
CROP = (88, 38, 1986, 1936)  # (x0, y0, x1, y1) -> 1898x1898
OUT_SIZE = 224


def preprocess_frame(frame: np.ndarray,
                     crop: tuple[int, int, int, int] = CROP,
                     size: int = OUT_SIZE) -> np.ndarray:
    """Crop to the square ROI, then downscale to size x size.

    Args:
        frame: HxWx3 RGB uint8.
    Returns:
        size x size x 3 RGB uint8.
    """
    x0, y0, x1, y1 = crop
    cropped = frame[y0:y1, x0:x1]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


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


def process_video(args):
    """Decode one video and return all encoded frames."""
    video_path, crop, size, fmt, jpeg_quality, batch = args
    stem = Path(video_path).stem
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        n = len(vr)
        if n == 0:
            return stem, [], 0, "empty video"
        x0, y0, x1, y1 = crop
        encoded: list[bytes] = []
        for start in range(0, n, batch):
            ids = list(range(start, min(start + batch, n)))
            frames = vr.get_batch(ids).asnumpy()
            if start == 0:
                h, w = frames[0].shape[:2]
                if y1 > h or x1 > w:
                    return (
                        stem, [], 0,
                        f"crop {crop} exceeds frame {w}x{h}",
                    )
            for fr in frames:
                out = preprocess_frame(fr, crop=crop, size=size)
                encoded.append(encode_frame(out, fmt, jpeg_quality))
        if len(encoded) != n:
            return (
                stem, [], 0,
                f"encoded {len(encoded)} frames, source has {n}",
            )
        return stem, encoded, n, None
    except Exception as e:  # noqa: BLE001 - keep worker alive, report per-video
        return stem, [], 0, f"{type(e).__name__}: {e}"


def estimate_map_size(n_videos: int, fmt: str, jpeg_quality: int) -> int:
    """Conservative LMDB map_size in bytes."""
    # Colour 224x224 frames; Pangaea days are typically ~700-800 frames.
    per_frame = {
        ("png", 0): 50_000,
        ("jpg", 70): 8_000,
        ("jpg", 75): 9_000,
        ("jpg", 80): 11_000,
        ("jpg", 85): 13_000,
        ("jpg", 90): 16_000,
    }
    bpf = per_frame.get((fmt, jpeg_quality if fmt == "jpg" else 0), 50_000)
    frames_per_video = 800
    # 1.4x headroom for LMDB/pickle overhead and unusually long videos.
    return int(n_videos * frames_per_video * bpf * 1.4) + (2 << 30)


def discover_videos(video_dir: Path) -> list[Path]:
    """Find MP4s either flat or nested under year/month/."""
    nested = sorted(video_dir.glob("[0-9]*/[0-9]*/*.mp4"))
    if nested:
        return nested
    return sorted(video_dir.glob("*.mp4"))


LEN_PREFIX = b"__len__:"
META_KEY_PREFIX = b"__"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"


def _len_key(stem: str) -> bytes:
    return LEN_PREFIX + stem.encode("utf-8")


def _is_payload_key(key: bytes) -> bool:
    return not key.startswith(META_KEY_PREFIX)


def _format_from_frame(buf: bytes) -> str | None:
    if buf.startswith(PNG_MAGIC):
        return "png"
    if buf.startswith(JPEG_MAGIC):
        return "jpg"
    return None


def stored_payload_format(env: lmdb.Environment) -> str | None:
    """Encoding of an existing store, from ``__meta__`` or the first frame."""
    with env.begin(write=False) as txn:
        raw_meta = txn.get(b"__meta__")
        if raw_meta:
            try:
                fmt = json.loads(raw_meta).get("format")
                if fmt in ("png", "jpg"):
                    return fmt
            except json.JSONDecodeError:
                pass
        cursor = txn.cursor()
        for key in cursor.iternext(keys=True, values=False):
            if not _is_payload_key(key):
                continue
            payload = txn.get(key)
            if not payload:
                continue
            frames = pickle.loads(payload)
            if not frames:
                continue
            return _format_from_frame(frames[0])
    return None


def load_completed(env: lmdb.Environment) -> tuple[dict[str, int], int]:
    """Return {stem: n_frames} and approximate payload bytes already stored.

    Uses the small ``__len__:{stem}`` keys and key-only cursor iteration so a
    PNG resume does not copy 20 GiB of frame payloads into Python. Falls back
    to unpickling a payload only when a day has frames but no ``__len__`` key.
    """
    video_lengths: dict[str, int] = {}
    payload_stems: set[str] = set()
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key in cursor.iternext(keys=True, values=False):
            if key.startswith(LEN_PREFIX):
                stem = key[len(LEN_PREFIX):].decode("utf-8")
                video_lengths[stem] = pickle.loads(txn.get(key))
            elif _is_payload_key(key):
                payload_stems.add(key.decode("utf-8"))
        for stem in [s for s in video_lengths if s not in payload_stems]:
            del video_lengths[stem]
        for stem in payload_stems - set(video_lengths):
            payload = txn.get(stem.encode("utf-8"))
            video_lengths[stem] = len(pickle.loads(payload))
        stat = txn.stat()
        bytes_total = (
            (stat["overflow_pages"] + stat["leaf_pages"]) * stat["psize"]
        )
    return video_lengths, bytes_total


def write_build_state(out_dir: Path, crop: tuple[int, int, int, int],
                      size: int, fmt: str, jpeg_quality: int) -> None:
    state = {
        "crop": list(crop),
        "out_size": size,
        "format": fmt,
        "jpeg_quality": jpeg_quality if fmt == "jpg" else None,
    }
    (out_dir / "build_state.json").write_text(json.dumps(state, indent=2))


def load_build_state(out_dir: Path) -> dict | None:
    path = out_dir / "build_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_lmdb(
    video_dir: Path,
    out_dir: Path,
    crop: tuple[int, int, int, int],
    size: int,
    fmt: str,
    jpeg_quality: int,
    workers: int,
    limit: int | None,
    map_size: int | None,
    resume: bool,
    decode_batch: int,
) -> None:
    all_video_paths = discover_videos(video_dir)
    if not all_video_paths:
        raise FileNotFoundError(f"No mp4s in {video_dir}")
    video_paths = all_video_paths
    if limit is not None:
        video_paths = video_paths[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = out_dir / "frames.lmdb"
    has_existing_data = lmdb_path.exists() and any(lmdb_path.iterdir())
    if has_existing_data and not resume:
        raise FileExistsError(
            f"{lmdb_path} already exists and is non-empty. "
            "Delete it first, choose another --out, or pass --resume."
        )
    lmdb_path.mkdir(parents=True, exist_ok=True)

    if map_size is None:
        map_size = estimate_map_size(len(video_paths), fmt, jpeg_quality)
    data_mdb = lmdb_path / "data.mdb"
    if has_existing_data and data_mdb.exists():
        # Never open smaller than the file already on disk.
        map_size = max(map_size, data_mdb.stat().st_size + (1 << 30))

    x0, y0, x1, y1 = crop
    print(f"Videos     : {len(video_paths)} from {video_dir}")
    print(f"Output     : {lmdb_path}")
    print(f"Preprocess : crop=({x0},{y0},{x1},{y1}) "
          f"-> {x1 - x0}x{y1 - y0} -> {size}x{size}")
    print(f"Encoding   : {fmt}"
          + (f" q={jpeg_quality}" if fmt == "jpg" else ""))
    print(f"Workers    : {workers}")
    print(f"decode_batch: {decode_batch}")
    print(f"map_size   : {map_size / (1 << 30):.1f} GiB")

    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    try:
        _build_into_env(
            env=env,
            out_dir=out_dir,
            video_dir=video_dir,
            video_paths=video_paths,
            crop=crop,
            size=size,
            fmt=fmt,
            jpeg_quality=jpeg_quality,
            workers=workers,
            has_existing_data=has_existing_data,
            decode_batch=decode_batch,
            x0=x0, y0=y0, x1=x1, y1=y1,
        )
    finally:
        env.close()


def _grow_map_size(env: lmdb.Environment, factor: float = 1.5) -> int:
    current = env.info()["map_size"]
    new_size = int(current * factor) + (1 << 30)
    env.set_mapsize(new_size)
    print(f"Grew map_size {current / (1 << 30):.1f} -> {new_size / (1 << 30):.1f} GiB")
    return new_size


def _build_into_env(
    env: lmdb.Environment,
    out_dir: Path,
    video_dir: Path,
    video_paths: list[Path],
    crop: tuple[int, int, int, int],
    size: int,
    fmt: str,
    jpeg_quality: int,
    workers: int,
    has_existing_data: bool,
    decode_batch: int,
    x0: int, y0: int, x1: int, y1: int,
) -> None:
    video_lengths: dict[str, int] = {}
    bytes_total = 0
    if has_existing_data:
        stored_fmt = stored_payload_format(env)
        state = load_build_state(out_dir)
        if state is not None:
            if tuple(state["crop"]) != crop or int(state["out_size"]) != size:
                raise ValueError(
                    f"Resume crop/size {state['crop']}/{state['out_size']} "
                    f"does not match this run {list(crop)}/{size}"
                )
            stored_fmt = stored_fmt or state.get("format")
        if stored_fmt is not None and stored_fmt != fmt:
            raise ValueError(
                f"Existing LMDB is {stored_fmt}, but this run is {fmt}. "
                "Pass matching --format or a new --out."
            )
        t_load = time.time()
        video_lengths, bytes_total = load_completed(env)
        completed = set(video_lengths)
        video_paths = [
            path for path in video_paths if path.stem not in completed]
        print(
            f"Resume     : {len(completed)} videos complete, "
            f"{len(video_paths)} remaining "
            f"({time.time() - t_load:.1f}s to index)"
        )

    write_build_state(out_dir, crop, size, fmt, jpeg_quality)

    worker_args = [
        (str(p), crop, size, fmt, jpeg_quality, decode_batch)
        for p in video_paths
    ]
    errors: list[tuple[str, str]] = []
    n_frames_total = sum(video_lengths.values())
    n_source_frames_total = n_frames_total
    t0 = time.time()

    pending: list[tuple[str, list[bytes]]] = []

    def flush(pending_items) -> None:
        nonlocal bytes_total
        if not pending_items:
            return
        payload_by_stem = []
        for stem, frames in pending_items:
            payload_by_stem.append((
                stem,
                pickle.dumps(frames, protocol=pickle.HIGHEST_PROTOCOL),
                len(frames),
            ))
        for attempt in range(8):
            try:
                added = 0
                with env.begin(write=True) as txn:
                    for stem, payload, n_frames in payload_by_stem:
                        txn.put(stem.encode("utf-8"), payload)
                        txn.put(
                            _len_key(stem),
                            pickle.dumps(
                                n_frames, protocol=pickle.HIGHEST_PROTOCOL),
                        )
                        added += len(payload)
                    txn.put(
                        b"__keys__",
                        pickle.dumps(
                            sorted(video_lengths),
                            protocol=pickle.HIGHEST_PROTOCOL,
                        ),
                    )
                    txn.put(
                        b"__videos__",
                        pickle.dumps(
                            video_lengths, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                bytes_total += added
                break
            except lmdb.MapFullError:
                if attempt == 7:
                    raise
                _grow_map_size(env)
        env.sync()
        pending_items.clear()

    if worker_args:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_video, a): a[0] for a in worker_args}
            with tqdm(total=len(futures), desc="videos", unit="vid") as pbar:
                for fut in as_completed(futures):
                    stem, frames, n_source, err = fut.result()
                    if err is not None:
                        errors.append((stem, err))
                        (out_dir / "errors.json").write_text(
                            json.dumps(errors, indent=2))
                        pbar.update(1)
                        continue
                    video_lengths[stem] = len(frames)
                    n_frames_total += len(frames)
                    n_source_frames_total += n_source
                    pending.append((stem, frames))
                    flush(pending)
                    pbar.set_postfix(
                        frames=n_frames_total,
                        GiB=f"{bytes_total / (1 << 30):.1f}",
                        err=len(errors),
                    )
                    pbar.update(1)

    flush(pending)

    keys = sorted(video_lengths)
    meta = {
        "dataset": "pangaea",
        "source": str(video_dir.resolve()),
        "crop": list(crop),
        "out_size": size,
        "format": fmt,
        "jpeg_quality": jpeg_quality if fmt == "jpg" else None,
        "n_videos": len(keys),
        "n_frames": n_frames_total,
        "n_source_frames": n_source_frames_total,
        "n_errors": len(errors),
        "layout": "video_payload_v1",
        "interpolation": "cv2.INTER_AREA",
        "pipeline": (
            f"crop[{x0}:{x1},{y0}:{y1}] -> "
            f"resize_{size}x{size}"
        ),
    }
    with env.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
        txn.put(
            b"__videos__",
            pickle.dumps(video_lengths, protocol=pickle.HIGHEST_PROTOCOL),
        )
        txn.put(b"__meta__", json.dumps(meta).encode("utf-8"))
    env.sync()

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    with open(out_dir / "videos.txt", "w") as f:
        for stem in keys:
            f.write(f"{stem} 0 {video_lengths[stem]} 0\n")
    if errors:
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=2))

    dt = time.time() - t0
    print(f"Done in {dt / 60:.1f} min")
    print(f"Wrote {len(keys)} videos / {n_frames_total} frames "
          f"({bytes_total / (1 << 30):.2f} GiB payload)")
    if errors:
        print(f"WARNING: {len(errors)} videos failed; see {out_dir / 'errors.json'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--x0", type=int, default=CROP[0])
    ap.add_argument("--y0", type=int, default=CROP[1])
    ap.add_argument("--x1", type=int, default=CROP[2])
    ap.add_argument("--y1", type=int, default=CROP[3])
    ap.add_argument("--size", type=int, default=OUT_SIZE)
    ap.add_argument(
        "--format", choices=("png", "jpg"), default="jpg",
        help="Frame encoding inside LMDB. JPG q75 is ~2.2 GB for the full "
             "Pangaea set; PNG is lossless, about 22 GB.",
    )
    ap.add_argument("--jpeg-quality", type=int, default=75)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--decode-batch", type=int, default=64,
        help="Frames decoded per Decord get_batch call. Smaller uses less "
             "RAM (8 is ~100 MB/worker vs ~830 MB at 64).",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N videos (smoke test).")
    ap.add_argument("--map-size", type=int, default=None,
                    help="LMDB map_size in bytes (auto if omitted).")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue a partially built LMDB in --out, skipping videos "
             "that already have a payload.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    crop = (args.x0, args.y0, args.x1, args.y1)
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError(f"Invalid crop box: {crop}")
    if args.decode_batch < 1:
        raise ValueError("--decode-batch must be >= 1")
    build_lmdb(
        video_dir=args.video_dir,
        out_dir=args.out,
        crop=crop,
        size=args.size,
        fmt=args.format,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
        limit=args.limit,
        map_size=args.map_size,
        resume=args.resume,
        decode_batch=args.decode_batch,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
