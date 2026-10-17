#!/usr/bin/env python
"""Draw the sun-blocker pictures.

  marked     the five notebook frames, plain next to the detections, with the
             hand-marked blocker pixels the detector missed in green
  day        frames spread over one day, plain next to the detections
  predicted  where the camera model puts the sun on the notebook frames

Run from models/VideoMAEv2, e.g.

  python tools/sun_blocker/render_figures.py marked day predicted \\
      --gif ../../figures/uoh_2016-06-15.gif --out_dir ../../figures
  python tools/sun_blocker/render_figures.py day \\
      --lmdb_path .../frames.lmdb --stem camera7_2019-08-02 --out_dir sun_blocker_vis
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.sun_blocker import SkyRegion, SunBlockerDetector  # noqa: E402

from camera import JUNE_2016, CameraModel  # noqa: E402
from frames import (BLOCKER, blocker_masks, gif_frames, lmdb_day,  # noqa: E402
                    notebook_frames)
from sunpos import sun_position  # noqa: E402

S, P, Z = 224, 16, 3
W, TITLE, GAP, LEGEND = S * Z, 70, 12, 120
RED, GREEN, YELLOW, CYAN = (235, 40, 40), (60, 230, 90), (255, 214, 0), (0, 210, 255)
BG, TEXT, DIM = (24, 24, 24), (240, 240, 240), (190, 190, 190)
NOTES = {
    '2016-05-04': 'clear sky, sun glare',
    '2017-01-17': 'clear sky, low sun (blocker not visible)',
    '2018-03-16': 'broken cloud',
    '2020-03-07': 'overcast, sun glare',
    '2025-04-30': 'after the 2020 cutoff: never masked',
}


def font(size, bold=False):
    names = (['arialbd.ttf', 'DejaVuSans-Bold.ttf'] if bold else
             ['arial.ttf', 'DejaVuSans.ttf'])
    for name in names:
        for folder in ('C:/Windows/Fonts', '/usr/share/fonts/truetype/dejavu'):
            try:
                return ImageFont.truetype(f'{folder}/{name}', size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


BIG, SMALL = font(26, bold=True), font(22)


def upscale(mask):
    return np.repeat(np.repeat(mask, Z, 0), Z, 1)


def panel(img, title, subtitle):
    tile = Image.new('RGB', (W, W + TITLE), BG)
    tile.paste(img, (0, TITLE))
    d = ImageDraw.Draw(tile)
    d.text((12, 6), title, font=BIG, fill=TEXT)
    d.text((12, 38), subtitle, font=SMALL, fill=DIM)
    return tile, d


def detection_pair(frame, det, title, marked=None):
    """(plain, detections) panels for one frame; ``marked`` = hand-marked
    blocker mask or None."""
    sky = det.sky_region.mask(S, S)
    edge = np.zeros_like(sky)
    edge[1:] |= sky[1:] != sky[:-1]
    edge[:, 1:] |= sky[:, 1:] != sky[:, :-1]
    found = det.dark_pixels(frame[None])[0]
    flagged = found.reshape(S // P, P, S // P, P).any(axis=(1, 3))
    base = Image.fromarray(frame).resize((W, W), Image.BICUBIC)
    plain, _ = panel(base, title, 'as stored (224x224), no overlay')
    img = np.asarray(base).astype(np.float32)
    layers = [(found, RED, 0.75), (edge, CYAN, 0.9)]
    if marked is not None:
        layers.insert(1, (marked & ~found, GREEN, 0.85))
    for mask, color, alpha in layers:
        m = upscale(mask)
        img[m] = (1 - alpha) * img[m] + alpha * np.array(color)
    if marked is not None and marked.any():
        truth = marked.reshape(S // P, P, S // P, P).any(axis=(1, 3))
        stats = (f"blocker patches found {(flagged & truth).sum()}/"
                 f"{truth.sum()}   ·   other patches flagged "
                 f"{(flagged & ~truth).sum()}")
    else:
        stats = f"patches flagged {flagged.sum()} of {(S // P)**2}"
    right, d = panel(Image.fromarray(img.round().astype(np.uint8)),
                     'detections', stats)
    for i, j in zip(*np.nonzero(flagged)):
        x0, y0 = j * P * Z, TITLE + i * P * Z
        d.rectangle([x0, y0, x0 + P * Z - 1, y0 + P * Z - 1], outline=YELLOW,
                    width=2)
    row = Image.new('RGB', (2 * W + GAP, W + TITLE), (10, 10, 10))
    row.paste(plain, (0, 0))
    row.paste(right, (W + GAP, 0))
    return row


def legend(title, items, width):
    strip = Image.new('RGB', (width, LEGEND), BG)
    d = ImageDraw.Draw(strip)
    d.text((16, 12), title, font=BIG, fill=TEXT)
    x = 16
    for color, kind, text in items:
        y = 66
        if kind == 'fill':
            d.rectangle([x, y, x + 34, y + 28], fill=color)
        elif kind == 'box':
            d.rectangle([x, y, x + 34, y + 28], outline=color, width=3)
        else:
            d.line([x, y + 14, x + 34, y + 14], fill=color, width=4)
        d.text((x + 46, y + 1), text, font=SMALL, fill=(225, 225, 225))
        x += 46 + int(d.textlength(text, font=SMALL)) + 40
    return strip


def stack(top, rows):
    sheet = Image.new('RGB', (top.width, LEGEND + len(rows) *
                              (rows[0].height + GAP)), (10, 10, 10))
    sheet.paste(top, (0, 0))
    for k, row in enumerate(rows):
        sheet.paste(row, (0, LEGEND + GAP + k * (row.height + GAP)))
    return sheet


def utc(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime('%H:%M')


def render_marked(det, out):
    rows = []
    for date, (frame, _) in notebook_frames().items():
        disc, arm = blocker_masks(date, frame)
        rows.append(detection_pair(frame, det, f"{date}  ·  {NOTES[date]}",
                                   marked=disc | arm))
    top = legend(f"Sun-blocker detector (contrast {det.contrast:.2f}, kernel "
                 f"{det.kernel} px)",
                 [(RED, 'fill', 'detected blocker pixels'),
                  (YELLOW, 'box', 'patch hidden from the model'),
                  (GREEN, 'fill', 'marked by hand, missed'),
                  (CYAN, 'line', 'sky region edge')], rows[0].width)
    stack(top, rows).save(out)


def render_day(det, frames, times, name, out, count=6):
    rows = [
        detection_pair(frames[i], det, f"{name}  ·  {utc(times[i])} UTC")
        for i in np.linspace(0, len(frames) - 1, count).astype(int)
    ]
    top = legend(f"{name}: contrast {det.contrast:.2f}, kernel {det.kernel} px",
                 [(RED, 'fill', 'detected blocker pixels'),
                  (YELLOW, 'box', 'patch hidden from the model'),
                  (CYAN, 'line', 'sky region edge')], rows[0].width)
    stack(top, rows).save(out)


def crosshair(d, x, y, color, r=7.5 * Z):
    d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
    for a, b in (((x - 2 * r, y), (x - r - 4, y)), ((x + r + 4, y),
                                                     (x + 2 * r, y)),
                 ((x, y - 2 * r), (x, y - r - 4)), ((x, y + r + 4),
                                                    (x, y + 2 * r))):
        d.line([a, b], fill=color, width=3)


def render_predicted(model, out):
    tiles = []
    for date, (frame, t) in notebook_frames().items():
        tile, d = panel(Image.fromarray(frame).resize((W, W), Image.BICUBIC),
                        f"{date} {utc(t)} UTC", '')
        az, zen = sun_position(t)
        x, y = model.project(az, zen) * Z
        crosshair(d, x, y + TITLE, YELLOW)
        if date in BLOCKER:
            cx, cy, _ = BLOCKER[date][0]
            e = np.hypot(x / Z - cx, y / Z - cy)
            sub = f"{e:.1f} px from the disc"
        elif date == '2025-04-30':
            # No blocker by then; the camera clock reads an hour behind UTC.
            x2, y2 = model.sun_pixel(t + 3600) * Z
            crosshair(d, x2, y2 + TITLE, CYAN)
            sub = 'cyan: clock + 1 h, on the sun'
        else:
            sub = f"sun azimuth {az:.0f}°, elevation {90 - zen:.0f}°"
        d.text((12, 38), sub, font=SMALL, fill=DIM)
        tiles.append(tile)
    sheet = Image.new('RGB', (3 * W + 2 * GAP, 2 * (W + TITLE) + GAP),
                      (10, 10, 10))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 3) * (W + GAP), (i // 3) * (W + TITLE + GAP)))
    d = ImageDraw.Draw(sheet)
    x0, y0 = 2 * (W + GAP) + 24, W + TITLE + GAP + 40
    for k, text in enumerate([
            'Yellow: where the sun (and so the blocker',
            'disc) should be, from the timestamp and the',
            'observatory location through the camera model.', '',
            f'centre ({model.u0:.1f}, {model.v0:.1f}) px, '
            f'north {model.north:.1f}°',
            f'r = {model.f:.1f}·zen {model.k:+.2f}·zen³ px (zen in rad)'
    ]):
        d.text((x0, y0 + 36 * k), text, font=SMALL, fill=(225, 225, 225))
    sheet.save(out)


def main():
    parser = argparse.ArgumentParser('Draw the sun-blocker pictures')
    parser.add_argument('figures', nargs='+',
                        choices=['marked', 'day', 'predicted'])
    parser.add_argument('--gif', type=str, default=None)
    parser.add_argument('--lmdb_path', type=str, default=None)
    parser.add_argument('--stem', type=str, default=None)
    parser.add_argument('--contrast', type=float, default=0.5)
    parser.add_argument('--kernel', type=int, default=15)
    parser.add_argument('--model', type=str, default=None,
                        help='camera model JSON (default: June 2016 fit)')
    parser.add_argument('--out_dir', type=str, default='.')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det = SunBlockerDetector(2, 2, P, contrast=args.contrast,
                             kernel=args.kernel, sky_region=SkyRegion())
    if 'marked' in args.figures:
        render_marked(det, out_dir / 'sun_blocker_side_by_side.png')
    if 'day' in args.figures:
        if args.gif:
            frames, times = gif_frames(args.gif)
            name, out = '2016-06-15', 'sun_blocker_june_2016.png'
        elif args.lmdb_path:
            frames, times = lmdb_day(args.lmdb_path, args.stem)
            name, out = args.stem, f'sun_blocker_{args.stem}.png'
        else:
            raise SystemExit('day needs --gif or --lmdb_path and --stem')
        render_day(det, frames, times, name, out_dir / out)
    if 'predicted' in args.figures:
        model = CameraModel.load(args.model) if args.model else JUNE_2016
        render_predicted(model, out_dir / 'sun_blocker_predicted_position.png')
    print(f"wrote {', '.join(args.figures)} to {out_dir.resolve()}")


if __name__ == '__main__':
    main()
