#!/usr/bin/env python
"""Build Pangaea finetune clip-index parquet files.

Each row is one 16-frame window already stored in the daily LMDB (no new
images). Times come from sidecar ``original_file`` names. Labels come from
``pv_output_valid.pkl``.

Window:
  frames  start, start+1, ..., start+15   (16 consecutive stored frames)
  starts every ``--stride`` frames (default 2 minutes)
  pv_log  = PV at those 16 times
  pv_pred = PV 15 minutes after the last frame

A window is kept only if:
  * all 16 frames exist in the LMDB day
  * consecutive frame times are ~60 s apart
  * all 16 PV values and the +15 min target exist in the pickle
    (nearest sample within ``--max-pv-offset-seconds``)

Day split: last ``--test-fraction`` of unique days (chronological) are test;
the rest are trainval. ``PVRegressionDataset`` then does the usual 90/10
day-block holdout on trainval.

Example
-------
  python build_pangaea_finetune_index.py
  python build_pangaea_finetune_index.py --limit-days 3
  python build_pangaea_finetune_index.py --lmdb data/pangaea_lmdb_224/frames.lmdb

Finetune (from scratch, no --finetune) then points --data_path at --out and
--lmdb_path at the same LMDB used here. See
models/VideoMAEv2/run_pangaea_lmdb_finetune.sh.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = SCRIPT_DIR / "data" / "raw_data" / "pangaea" / "PVotSky"
DEFAULT_LMDB = SCRIPT_DIR / "data" / "pangaea_lmdb_224_png" / "frames.lmdb"
DEFAULT_PV = (
    SCRIPT_DIR / "data" / "raw_data" / "pangaea" / "PVotSky"
    / "pv_data" / "pv_output_valid.pkl"
)
DEFAULT_OUT = SCRIPT_DIR / "data" / "pangaea_finetune"


def parse_original_file(name: str) -> datetime:
    """``20220801064501_00160.jpg`` -> 2022-08-01 06:45:01."""
    token = Path(name).stem.split("_")[0]
    return datetime.strptime(token, "%Y%m%d%H%M%S")


def sidecar_for_stem(video_dir: Path, stem: str) -> Path | None:
    year, month = stem[:4], stem[4:6]
    nested = video_dir / year / month / f"{stem}.json"
    if nested.exists():
        return nested
    flat = video_dir / f"{stem}.json"
    return flat if flat.exists() else None


def load_lmdb_lengths(lmdb_path: Path) -> dict[str, int]:
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False)
    try:
        with env.begin(write=False) as txn:
            stored = txn.get(b"__videos__")
            if stored is not None:
                return pickle.loads(stored)
            stored_keys = txn.get(b"__keys__")
            if stored_keys is None:
                raise FileNotFoundError(
                    f"{lmdb_path} has no __videos__ / __keys__; is it finished?"
                )
            keys = pickle.loads(stored_keys)
            lengths = {}
            for stem in keys:
                payload = txn.get(stem.encode("utf-8"))
                if payload is None:
                    continue
                lengths[stem] = len(pickle.loads(payload))
            return lengths
    finally:
        env.close()


def lookup_pv(pv_ns: np.ndarray, pv_values: np.ndarray,
              ts: datetime, max_offset: float) -> float | None:
    """Nearest PV sample to ``ts`` if it is within ``max_offset`` seconds."""
    t_ns = int(pd.Timestamp(ts).value)
    i = int(np.searchsorted(pv_ns, t_ns))
    best = None
    best_dt = max_offset + 1.0
    for j in (i - 1, i):
        if 0 <= j < len(pv_ns):
            dt = abs(int(pv_ns[j]) - t_ns) / 1e9
            if dt < best_dt:
                best_dt = dt
                best = j
    if best is None or best_dt > max_offset:
        return None
    return float(pv_values[best])


def consecutive_minutes(times: list[datetime], slop: float = 5.0) -> bool:
    for a, b in zip(times, times[1:]):
        if abs((b - a).total_seconds() - 60.0) > slop:
            return False
    return True


def windows_for_day(
    stem: str,
    n_frames: int,
    sidecar: dict,
    pv_ns: np.ndarray,
    pv_values: np.ndarray,
    clip_len: int,
    stride: int,
    horizon_minutes: int,
    max_pv_offset: float,
) -> list[dict]:
    frames_meta = sidecar.get("frames") or []
    n = min(n_frames, len(frames_meta))
    if n < clip_len:
        return []

    times: list[datetime] = []
    for i in range(n):
        times.append(parse_original_file(frames_meta[i]["original_file"]))

    horizon = timedelta(minutes=horizon_minutes)
    rows: list[dict] = []
    last_start = n - clip_len
    for start in range(0, last_start + 1, stride):
        clip_times = times[start:start + clip_len]
        if len(clip_times) != clip_len:
            continue
        if not consecutive_minutes(clip_times):
            continue
        pv_log = []
        ok = True
        for t in clip_times:
            val = lookup_pv(pv_ns, pv_values, t, max_pv_offset)
            if val is None:
                ok = False
                break
            pv_log.append(val)
        if not ok:
            continue
        target_t = clip_times[-1] + horizon
        pv_pred = lookup_pv(pv_ns, pv_values, target_t, max_pv_offset)
        if pv_pred is None:
            continue
        rows.append({
            "video_key": stem,
            "start_idx": int(start),
            "time": pd.Timestamp(clip_times[-1]),
            "times": [t.isoformat() for t in clip_times],
            "pv_log": np.asarray(pv_log, dtype=np.float32),
            "pv_pred": float(pv_pred),
            "cloudiness": 0.0,
        })
    return rows


def write_split(df: pd.DataFrame, out_path: Path, with_cloudiness: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["video_key", "time", "pv_log", "pv_pred"]
    if with_cloudiness:
        cols.append("cloudiness")
    cols.extend(["start_idx", "times"])
    df.loc[:, cols].to_parquet(out_path, index=False)


def build_index(
    video_dir: Path,
    lmdb_path: Path,
    pv_path: Path,
    out_dir: Path,
    clip_len: int,
    stride: int,
    horizon_minutes: int,
    max_pv_offset: float,
    test_fraction: float,
    limit_days: int | None,
) -> None:
    lengths = load_lmdb_lengths(lmdb_path)
    stems = sorted(lengths)
    if limit_days is not None:
        stems = stems[:limit_days]
    if not stems:
        raise FileNotFoundError(f"No LMDB videos in {lmdb_path}")

    pv = pd.read_pickle(pv_path)
    if not isinstance(pv, pd.Series):
        raise TypeError(f"Expected Series in {pv_path}, got {type(pv)}")
    pv = pv.sort_index()
    # datetime64[s] indexes store asi8 in seconds; Timestamp.value is ns.
    pv_ns = np.asarray(
        pd.DatetimeIndex(pv.index).astype("datetime64[ns]").asi8,
        dtype=np.int64,
    )
    pv_values = pv.to_numpy(dtype=np.float64)

    all_rows: list[dict] = []
    skipped_no_sidecar = 0
    for stem in tqdm(stems, desc="days"):
        sidecar_path = sidecar_for_stem(video_dir, stem)
        if sidecar_path is None:
            skipped_no_sidecar += 1
            continue
        sidecar = json.loads(sidecar_path.read_text())
        all_rows.extend(
            windows_for_day(
                stem,
                lengths[stem],
                sidecar,
                pv_ns,
                pv_values,
                clip_len=clip_len,
                stride=stride,
                horizon_minutes=horizon_minutes,
                max_pv_offset=max_pv_offset,
            )
        )

    if not all_rows:
        raise RuntimeError("No valid clips; check PV alignment and sidecars.")

    df = pd.DataFrame(all_rows)
    days = sorted(df["video_key"].unique())
    n_test_days = max(1, int(round(len(days) * test_fraction)))
    test_days = set(days[-n_test_days:])
    test_df = df[df["video_key"].isin(test_days)].reset_index(drop=True)
    trainval_df = df[~df["video_key"].isin(test_days)].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "metadata_trainval" / "metadata_with_cloudiness.parquet"
    test_path = out_dir / "metadata_test" / "metadata.parquet"
    write_split(trainval_df, train_path, with_cloudiness=True)
    write_split(test_df, test_path, with_cloudiness=False)

    meta = {
        "dataset": "pangaea",
        "lmdb_path": str(lmdb_path.resolve()),
        "pv_path": str(pv_path.resolve()),
        "video_dir": str(video_dir.resolve()),
        "clip_len": clip_len,
        "stride": stride,
        "horizon_minutes": horizon_minutes,
        "max_pv_offset_seconds": max_pv_offset,
        "n_lmdb_days": len(stems),
        "n_days_with_clips": int(df["video_key"].nunique()),
        "skipped_no_sidecar": skipped_no_sidecar,
        "n_clips": int(len(df)),
        "n_trainval": int(len(trainval_df)),
        "n_test": int(len(test_df)),
        "n_trainval_days": int(trainval_df["video_key"].nunique()),
        "n_test_days": int(test_df["video_key"].nunique()),
        "test_days": sorted(test_days),
        "test_day_start": min(test_days),
        "test_day_end": max(test_days),
        "layout": "lmdb_clip_index_v1",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(trainval_df)} trainval clips "
          f"({meta['n_trainval_days']} days) -> {train_path}")
    print(f"Wrote {len(test_df)} test clips "
          f"({meta['n_test_days']} days, {meta['test_day_start']}.."
          f"{meta['test_day_end']}) -> {test_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--lmdb", type=Path, default=DEFAULT_LMDB,
                    help="Finished daily LMDB (used for day keys / frame counts). "
                         "Default is the PNG store.")
    ap.add_argument("--pv", type=Path, default=DEFAULT_PV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--horizon-minutes", type=int, default=15)
    ap.add_argument("--max-pv-offset-seconds", type=float, default=30.0)
    ap.add_argument("--test-fraction", type=float, default=0.1)
    ap.add_argument("--limit-days", type=int, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_index(
        video_dir=args.video_dir,
        lmdb_path=args.lmdb,
        pv_path=args.pv,
        out_dir=args.out,
        clip_len=args.clip_len,
        stride=args.stride,
        horizon_minutes=args.horizon_minutes,
        max_pv_offset=args.max_pv_offset_seconds,
        test_fraction=args.test_fraction,
        limit_days=args.limit_days,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
