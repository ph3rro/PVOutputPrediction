"""Sun-blocker masking for UoH (University of Hertfordshire) pretraining clips.

Until 2020-07-01 the UoH all-sky camera carried a sun blocker: a black disc on
a thin black arm reaching in from the edge of the frame. Its pixels are near
black, so they are located with a gray-level threshold on the augmented clip.
Every tubelet (tubelet_size frames x patch_size x patch_size pixels) that
contains such pixels is

  * forced into the encoder mask, so the encoder never sees it, and
  * excluded from the decoder's reconstruction loss, so the model is never
    trained to reconstruct it.

Whether a clip carries the blocker is decided from the date embedded in its
LMDB video stem (e.g. ``camera7_2019-06-20``).
"""
import datetime as dt
import re

import numpy as np
import torch

_DATE_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')


def parse_date(text):
    """Parse a ``YYYY-MM-DD`` string into a ``datetime.date``."""
    match = _DATE_RE.fullmatch(str(text).strip())
    if match is None:
        raise ValueError(f"Expected a YYYY-MM-DD date, got {text!r}")
    return dt.date(*(int(value) for value in match.groups()))


def date_from_stem(stem):
    """Return the ``YYYY-MM-DD`` date embedded in a video stem, or None."""
    match = _DATE_RE.search(stem)
    if match is None:
        return None
    try:
        return dt.date(*(int(value) for value in match.groups()))
    except ValueError:
        return None


def clip_has_sun_blocker(stem, until):
    """True when the stem is dated on or before ``until`` (a datetime.date)."""
    date = date_from_stem(stem)
    return date is not None and date <= until


class SunBlockerDetector:
    """Flag tubelets containing dark (sun-blocker) pixels.

    The detector runs on the augmented and normalised clip tensor of shape
    ``[num_frames * 3, H, W]`` that ``DataAugmentationForVideoMAEv2``
    produces, so it sees exactly the pixels the model sees (same random crop
    and resize) and its output lines up with the token grid.

    Args:
        num_frames: frames per clip (T).
        tubelet_size: frames per token (temporal patch size).
        patch_size: spatial patch size, int or (ph, pw).
        threshold: gray level in [0, 255]; pixels below it count as dark.
        min_pixels: a tubelet is flagged when it holds at least this many
            dark pixels.
        mean, std: per-channel normalisation used by the augmentation, so
            the detector can undo it.
    """

    def __init__(self,
                 num_frames,
                 tubelet_size,
                 patch_size,
                 threshold=60,
                 min_pixels=1,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if num_frames % tubelet_size != 0:
            raise ValueError(
                f"num_frames={num_frames} is not a multiple of "
                f"tubelet_size={tubelet_size}")
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must lie in [0, 255]")
        if min_pixels < 1:
            raise ValueError("min_pixels must be at least 1")
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.threshold = float(threshold) / 255.
        self.min_pixels = int(min_pixels)
        # Broadcast against [T, 3, H, W].
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def dark_pixels(self, process_data):
        """Per-pixel dark mask, shape ``[T, H, W]`` (bool)."""
        _, h, w = process_data.shape
        frames = process_data.view(self.num_frames, 3, h, w)
        gray = (frames * self.std + self.mean).mean(dim=1)  # [T, H, W] in [0, 1]
        return gray < self.threshold

    def __call__(self, process_data):
        """Tubelet mask of shape ``[T // tubelet_size, H // ph, W // pw]``.

        True marks a tubelet that contains at least ``min_pixels`` dark pixels.
        """
        dark = self.dark_pixels(process_data)
        t, h, w = dark.shape
        ph, pw = self.patch_size
        if h % ph or w % pw:
            raise ValueError(
                f"Frame size {h}x{w} is not divisible by patch size {ph}x{pw}")
        dark = dark.view(t // self.tubelet_size, self.tubelet_size, h // ph,
                         ph, w // pw, pw)
        counts = dark.sum(dim=(1, 3, 5))
        return counts >= self.min_pixels

    def __repr__(self):
        return (
            "SunBlockerDetector(threshold=%d/255, min_pixels=%d, "
            "tubelet_size=%d, patch_size=%s)" %
            (round(self.threshold * 255), self.min_pixels, self.tubelet_size,
             self.patch_size))


def summarize_flags(flags):
    """Return ``(n_flagged, n_total)`` for a boolean array of clip flags."""
    flags = np.asarray(flags, dtype=bool)
    return int(flags.sum()), int(flags.size)
