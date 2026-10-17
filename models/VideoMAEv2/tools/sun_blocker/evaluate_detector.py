#!/usr/bin/env python
"""Score the sun-blocker detector at several --sun_blocker_contrast values.

For each value it reports, on the five frames from the preprocessing notebook,
how many hand-marked blocker patches are flagged and how many other patches
are, and on a day of frames how many of the 196 patches are flagged per frame.

Run from models/VideoMAEv2, e.g.

  python tools/sun_blocker/evaluate_detector.py --gif ../../figures/uoh_2016-06-15.gif
  python tools/sun_blocker/evaluate_detector.py \\
      --lmdb_path /home/ubuntu/PVOutputPrediction/preprocessing/data/uoh_lmdb_minute_224/frames.lmdb \\
      --stem camera7_2016-06-15
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.sun_blocker import SkyRegion, SunBlockerDetector  # noqa: E402

from frames import blocker_masks, gif_frames, lmdb_day, notebook_frames  # noqa: E402

S, P = 224, 16


def patches(mask):
    """[H, W] bool -> [H/16, W/16] bool, True where a patch holds any pixel."""
    return mask.reshape(S // P, P, S // P, P).any(axis=(1, 3))


def main():
    parser = argparse.ArgumentParser('Score the sun-blocker detector')
    parser.add_argument('--gif', type=str, default=None)
    parser.add_argument('--lmdb_path', type=str, default=None)
    parser.add_argument('--stem', type=str, default=None)
    parser.add_argument('--contrast', type=float, nargs='+',
                        default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    parser.add_argument('--kernel', type=int, default=15)
    args = parser.parse_args()

    det = SunBlockerDetector(2, 2, P, kernel=args.kernel,
                             sky_region=SkyRegion())
    sky = det.sky_region.mask(S, S)
    marked = []
    for date, (frame, _) in notebook_frames().items():
        disc, arm = blocker_masks(date, frame)
        marked.append((det.darkness(frame[None])[0] * sky, disc | arm))
    day = None
    if args.gif:
        day = gif_frames(args.gif)[0]
    elif args.lmdb_path:
        day = lmdb_day(args.lmdb_path, args.stem)[0]
    day_dark = None if day is None else det.darkness(day) * sky

    print(f"{'contrast':>8} | notebook frames: blocker patches, other patches"
          + (" | day: patches flagged per frame (mean / max of 196)"
             if day is not None else ''))
    for c in args.contrast:
        found = total = other = 0
        for dark, blocker in marked:
            flagged, truth = patches(dark > c), patches(blocker)
            found += (flagged & truth).sum()
            total += truth.sum()
            other += (flagged & ~truth).sum()
        line = f"{c:8.2f} | blocker {found:2d}/{total}, other {other:3d}"
        if day_dark is not None:
            per = np.array([patches(d > c).sum() for d in day_dark])
            line += f"{'':22} | {per.mean():5.1f} / {per.max():3d}"
        print(line)


if __name__ == '__main__':
    main()
