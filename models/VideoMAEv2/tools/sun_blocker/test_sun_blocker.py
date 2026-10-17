#!/usr/bin/env python
"""Checks for the sun-blocker detector and the camera tools.

Run from models/VideoMAEv2 with ``python tools/sun_blocker/test_sun_blocker.py``
(or pytest). Needs the packages the training code needs (lmdb, cv2, torch).
"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset.pretrain_datasets import DataAugmentationForVideoMAEv2  # noqa: E402
from dataset.sun_blocker import SkyRegion, SunBlockerDetector  # noqa: E402

from camera import JUNE_2016  # noqa: E402
from frames import BLOCKER, blocker_masks, notebook_frames, unix  # noqa: E402
from sunpos import sun_position  # noqa: E402

T, S, P = 4, 224, 16
YY, XX = np.mgrid[0:S, 0:S]


def make_args(**over):
    a = dict(input_size=S, mask_type='tube', mask_ratio=0.9,
             decoder_mask_ratio=0.0, decoder_mask_type='run_cell',
             window_size=(T // 2, S // P, S // P), num_frames=T,
             tubelet_size=2, patch_size=(P, P), sun_blocker_masking=True,
             sun_blocker_contrast=0.5, sun_blocker_kernel=15,
             sun_blocker_min_pixels=1,
             sun_blocker_circle=[0.502, 0.460, 0.472],
             sun_blocker_image_rows=[0.104, 0.896])
    a.update(over)
    return SimpleNamespace(**a)


def synthetic_frame():
    """Smooth sky with a disc, a dark and a faint arm, a wide dark patch and
    a disc lit gray by sun glare."""
    sky = SkyRegion().mask(S, S)
    g = 0.75 - 0.15 * np.hypot(XX - 112, YY - 103) / 106
    g = np.where(sky, g, 0.0)
    masks = {
        'disc': (XX - 140)**2 + (YY - 120)**2 <= 7**2,
        'arm': (YY == 120) & (XX > 147) & (XX < 205),
        'faint arm': (YY == 100) & (XX > 147) & (XX < 205),
        'wide dark sky': (XX >= 50) & (XX < 100) & (YY >= 60) & (YY < 110),
        'glare disc': np.hypot(XX - 80, YY - 160) <= 6,
    }
    g[masks['disc']] = 0.12
    g[masks['arm']] *= 0.4  # 60% darker than the sky: counts
    g[masks['faint arm']] *= 0.55  # 45% darker: below the 0.5 default
    g[masks['wide dark sky']] = 0.25
    g[np.hypot(XX - 80, YY - 160) <= 16] = 0.9
    g[np.hypot(XX - 80, YY - 160) <= 7] = 0.32
    rgb = (np.repeat(g[..., None], 3, -1) * 255).round().astype(np.uint8)
    return rgb, masks, sky


def test_sky_region():
    m = SkyRegion().mask(S, S)
    assert m[103, 112] and m[23, 112] and m[200, 112]
    assert not m[:23].any() and not m[201:].any()  # padding rows
    assert not m[110, 3] and not m[110, 220]  # camera rim
    assert not m[0, 0] and not m[-1, -1]


def test_detector_synthetic():
    rgb, masks, sky = synthetic_frame()
    det = SunBlockerDetector(T, 2, P, sky_region=SkyRegion())
    dark = det.dark_pixels(np.stack([rgb] * T))[0]
    assert dark[masks['disc']].mean() > 0.95
    assert dark[masks['arm']].mean() > 0.95
    assert dark[masks['glare disc']].mean() > 0.95
    assert dark[masks['faint arm']].mean() < 0.05
    assert dark[masks['wide dark sky']].mean() < 0.02
    assert not dark[~sky].any()
    tub = det.tubelets(det.dark_pixels(np.stack([rgb] * T)))
    assert tub.shape == (T // 2, S // P, S // P)
    assert tub[0, 120 // P, 140 // P] and not tub[0, 80 // P, 70 // P]
    for bad in (dict(kernel=14), dict(kernel=1), dict(contrast=0),
                dict(contrast=1), dict(min_pixels=0)):
        try:
            SunBlockerDetector(T, 2, P, **bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_masks_follow_crops():
    random.seed(0)
    np.random.seed(0)
    aug = DataAugmentationForVideoMAEv2(make_args())
    frame = np.full((S, S, 3), 191, np.uint8)
    frame[(XX - 140)**2 + (YY - 120)**2 <= 7**2] = 31
    clip = [Image.fromarray(frame)] * T
    mean = np.array(aug.input_mean).reshape(1, 3, 1, 1)
    std = np.array(aug.input_std).reshape(1, 3, 1, 1)
    for _ in range(100):
        process_data, blocker = aug.augment_with_blocker((clip, None))
        assert blocker.shape == (T, S, S) and blocker.dtype == bool
        gray = (process_data.view(T, 3, S, S).numpy() * std + mean).mean(1)
        truth = gray[0] < (0.12 + 0.75) / 2
        iou = (blocker[0] & truth).sum() / max((blocker[0] | truth).sum(), 1)
        assert iou > 0.9, iou


def test_transform_outputs():
    rgb, _, _ = synthetic_frame()
    clip = [Image.fromarray(rgb)] * T
    aug = DataAugmentationForVideoMAEv2(make_args())
    out = aug((clip, None), sun_blocker=True)
    assert len(out) == 4 and out[3].shape == out[1].shape
    assert np.all(out[1][:, out[3].any(0)] == 1)  # flagged = hidden
    assert not aug((clip, None), sun_blocker=False)[3].any()
    plain = DataAugmentationForVideoMAEv2(make_args(sun_blocker_masking=False))
    assert len(plain((clip, None))) == 3


def test_notebook_frames():
    """Defaults on the five notebook frames: 26/33 blocker patches found and
    30 other patches flagged on 2026-10-12."""
    det = SunBlockerDetector(2, 2, P, sky_region=SkyRegion())
    found = total = other = 0
    for date, (frame, _) in notebook_frames().items():
        disc, arm = blocker_masks(date, frame)
        flagged = det.dark_pixels(frame[None])[0]
        flagged = flagged.reshape(S // P, P, S // P, P).any(axis=(1, 3))
        truth = (disc | arm).reshape(S // P, P, S // P, P).any(axis=(1, 3))
        found += (flagged & truth).sum()
        total += truth.sum()
        other += (flagged & ~truth).sum()
    assert total == 33 and found >= 26 and other <= 30, (found, total, other)


def test_sun_position():
    # pvlib 0.16 NREL algorithm at the same site
    for ts, az, zen in (('2016-06-15 10:00:00', 129.281, 36.682),
                        ('2018-03-16 12:21:42', 183.947, 53.491),
                        ('2019-12-21 12:00:00', 180.406, 75.213)):
        a, z = sun_position(unix(ts))
        assert abs(a - az) < 0.05 and abs(z - zen) < 0.05, (ts, a, z)


def test_camera_model():
    """The June 2016 fit puts the sun on the 2016 and 2018 discs."""
    frames = notebook_frames()
    for date in ('2016-05-04', '2018-03-16'):
        cx, cy, _ = BLOCKER[date][0]
        e = np.hypot(*(JUNE_2016.sun_pixel(frames[date][1]) - (cx, cy)))
        assert e < 3.5, (date, e)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
