#!/usr/bin/env python
"""Fit the camera model that maps the sun's direction to a stored-frame pixel.

The sun-blocker disc sits on the sun, so following the disc through a day of
timestamped frames pairs computed sun directions with pixels. The script
tracks the disc, fits a fisheye camera.CameraModel, reports how far the model
is from the tracked disc, checks it on the hand-marked notebook frames from
other dates and looks for a clock offset in the day's timestamps.

Run from models/VideoMAEv2, e.g.

  # the June 15, 2016 GIF in figures/
  python tools/sun_blocker/calibrate_camera.py --gif ../../figures/uoh_2016-06-15.gif

  # days of the minute-aligned LMDB; --no_fit only checks the June 2016 model
  python tools/sun_blocker/calibrate_camera.py \\
      --lmdb_path /home/ubuntu/PVOutputPrediction/preprocessing/data/uoh_lmdb_minute_224/frames.lmdb \\
      --stem camera7_2016-06-15 --window 08:00 14:00
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.sun_blocker import SkyRegion  # noqa: E402

from camera import JUNE_2016, CameraModel  # noqa: E402
from frames import BLOCKER, gif_frames, lmdb_day, notebook_frames  # noqa: E402
from sunpos import sun_position  # noqa: E402

S = 224
SKY = SkyRegion().mask(S, S)


def get_args():
    parser = argparse.ArgumentParser('Calibrate the UoH camera on the sun')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--gif', type=str, help='June 15, 2016 timelapse GIF')
    source.add_argument('--lmdb_path', type=str,
                        help='minute-aligned UoH LMDB (frames.lmdb)')
    parser.add_argument('--stem', nargs='+', default=[],
                        help='video stems to use with --lmdb_path')
    parser.add_argument('--window', nargs=2, metavar=('FROM', 'TO'),
                        help='UTC times HH:MM to use from each day')
    parser.add_argument('--model', type=str, default=None,
                        help='camera model JSON to start from (default: '
                        'the June 2016 fit)')
    parser.add_argument('--start', nargs=2, type=float, metavar=('X', 'Y'),
                        help='disc position in the first frame (default: '
                        'where the starting model puts the sun)')
    parser.add_argument('--search', type=float, default=25,
                        help='search radius in px for the disc in the first '
                        'frame')
    parser.add_argument('--no_fit', action='store_true',
                        help='only check the starting model')
    parser.add_argument('--out', type=str, default='camera_model.json')
    return parser.parse_args()


def disc_core(frame):
    """Compact dark features (the disc, not the thin arm) as a bool mask."""
    gray = frame.astype(np.float32).mean(-1) / 255
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    dark = ((bg - gray) / np.maximum(bg, 0.2) > 0.35) & SKY
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN,
                            disc).astype(bool)


def track(frames, start, search=25, window=8, min_pixels=15):
    """Follow the disc from frame to frame, searching near its last position.

    A plain "darkest round blob" search latches onto blue gaps between
    clouds, so the disc is only looked for near where it was a minute
    earlier. Returns positions ``[N, 2]`` and a bool array of the frames
    where it was found (elsewhere the last position is carried over).
    """
    pos, found = [], []
    px, py = start
    radius = search
    for frame in frames:
        yy, xx = np.nonzero(disc_core(frame))
        near = np.hypot(xx + 0.5 - px, yy + 0.5 - py) <= radius
        ok = near.sum() >= min_pixels
        if ok:
            px, py = xx[near].mean() + 0.5, yy[near].mean() + 0.5
            radius = window
        pos.append((px, py))
        found.append(ok)
    return np.array(pos), np.array(found)


def fit(pixels, azimuth, zenith, init):
    """Least-squares camera model, robust to a few mistracked frames."""

    def residual(p):
        model = CameraModel(*p, mirror=init.mirror)
        return (model.project(azimuth, zenith) - pixels).ravel()

    best = None
    for north in init.north + np.arange(-180, 180, 30):
        p0 = [init.u0, init.v0, init.f, init.k, north]
        res = least_squares(residual, p0, loss='soft_l1', f_scale=1.0)
        if best is None or res.cost < best.cost:
            best = res
    u0, v0, f, k, north = best.x
    return CameraModel(u0, v0, f, k, north % 360, init.mirror)


def errors(model, pixels, times):
    return np.hypot(*(model.sun_pixel(times) - pixels).T)


def clock_offset(model, pixels, times):
    """Clock shift (minutes) that best fits the model, and its median error."""
    shifts = np.arange(-120, 121)
    err = [np.median(errors(model, pixels, times + 60 * m)) for m in shifts]
    i = int(np.argmin(err))
    return shifts[i], err[i]


def load_days(args):
    """[(name, frames, times)] from the GIF or the LMDB."""
    if args.gif:
        frames, times = gif_frames(args.gif)
        days = [('2016-06-15 (GIF)', frames, times)]
    else:
        if not args.stem:
            raise SystemExit('--lmdb_path needs --stem')
        days = [(stem, *lmdb_day(args.lmdb_path, stem)) for stem in args.stem]
    if args.window:
        lo, hi = ([int(v) for v in hm.split(':')] for hm in args.window)
        kept = []
        for name, frames, times in days:
            minute = (times % 86400) // 60
            keep = (minute >= lo[0] * 60 + lo[1]) & (minute <= hi[0] * 60 +
                                                      hi[1])
            kept.append((name, frames[keep], times[keep]))
        days = kept
    return days


def main():
    args = get_args()
    init = CameraModel.load(args.model) if args.model else JUNE_2016
    pixels, times = [], []
    for name, frames, t in load_days(args):
        if not len(frames):
            print(f"{name}: no frames in the window")
            continue
        start = args.start if args.start else init.sun_pixel(t[0])
        pos, found = track(frames, start, search=args.search)
        shift, err = clock_offset(init, pos[found], t[found])
        print(f"{name}: {len(frames)} frames, disc found in {found.sum()}; "
              f"starting model: median error "
              f"{np.median(errors(init, pos[found], t[found])):.1f} px, best "
              f"clock offset {shift:+d} min ({err:.1f} px)")
        pixels.append(pos[found])
        times.append(t[found])
    if not pixels:
        raise SystemExit('No disc positions to fit')
    pixels, times = np.concatenate(pixels), np.concatenate(times)

    model = init
    if not args.no_fit:
        model = fit(pixels, *sun_position(times), init)
        err = errors(model, pixels, times)
        print(f"fitted on {len(err)} frames: median error "
              f"{np.median(err):.2f} px, 90th percentile "
              f"{np.percentile(err, 90):.2f} px, max {err.max():.2f} px")
        print(f"model: {model}")
        model.save(args.out, frames=int(len(err)),
                   median_error=float(np.median(err)))
        print(f"saved {args.out}")

    print("check on the hand-marked notebook frames:")
    for date, (frame, t) in notebook_frames().items():
        if date in BLOCKER:
            cx, cy, _ = BLOCKER[date][0]
            e = np.hypot(*(model.sun_pixel(t) - (cx, cy)))
            print(f"  {date}: {e:.1f} px from the disc")


if __name__ == '__main__':
    main()
