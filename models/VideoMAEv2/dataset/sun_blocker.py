"""Sun-blocker masking for UoH (University of Hertfordshire) pretraining clips.

Until 2020-07-01 the UoH all-sky camera carried a sun blocker: a black disc on
a thin black arm reaching in from the edge of the sky. It is found by local
darkness: a pixel counts when it is much darker than the sky around it, which
picks out the disc and arm on clear, cloudy and overcast days alike while
leaving wide stretches of dark sky alone. Only pixels inside the sky region
count: the sky inside the camera's dark rim, minus the black rows that
build_uoh_lmdb.py pads above and below the camera image. Every tubelet
(tubelet_size frames x patch_size x patch_size pixels) that contains blocker
pixels is

  * forced into the encoder mask, so the encoder never sees it, and
  * excluded from the decoder's reconstruction loss, so the model is never
    trained to reconstruct it.

Whether a clip carries the blocker is decided from the date embedded in its
LMDB video stem (e.g. ``camera7_2019-06-20``).
"""
import datetime as dt
import re

import cv2
import numpy as np

_DATE_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')

# Sky region of UoH frames stored by build_uoh_lmdb.py, as fractions of the
# (square) stored frame. That script keeps x in [24, 630) of each 640x480
# camera frame and pads 63 black rows above and below, giving 606x606. The
# circle lies 5 px (at 224x224) inside the dark rim around the sky, a circle
# fitted on the camera7 frames shown in video_only_data_preprocess.ipynb. It
# reaches past the top and bottom of the camera image, so the padding rows are
# cut off separately.
UOH_SKY_CIRCLE = (0.502, 0.460, 0.472)  # centre x, centre y, radius
UOH_IMAGE_ROWS = (0.104, 0.896)  # top and bottom of the camera image


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


class SkyRegion:
    """Pixels of a stored frame in which the sun blocker can appear.

    The region is the sky circle intersected with the rows that hold camera
    image. Pixels outside it (the camera rim, the black corners beyond it and
    the black padding rows) never count as blocker.

    Args:
        circle: (centre x, centre y, radius); centre x and radius as fractions
            of the frame width, centre y as a fraction of its height.
        rows: (top, bottom) of the camera image as fractions of the frame
            height.
    """

    def __init__(self, circle=UOH_SKY_CIRCLE, rows=UOH_IMAGE_ROWS):
        cx, cy, radius = (float(value) for value in circle)
        top, bottom = (float(value) for value in rows)
        if radius <= 0:
            raise ValueError("circle radius must be positive")
        if not 0 <= top < bottom <= 1:
            raise ValueError("rows must satisfy 0 <= top < bottom <= 1")
        self.circle = (cx, cy, radius)
        self.rows = (top, bottom)
        self._masks = {}

    def mask(self, width, height):
        """Bool array ``[height, width]``, True for pixels inside the region.

        Cached per size; do not modify the returned array.
        """
        if (width, height) not in self._masks:
            cx, cy, radius = self.circle
            top, bottom = self.rows
            x = np.arange(width) + 0.5  # pixel centres
            y = np.arange(height) + 0.5
            inside = ((x[None, :] - cx * width)**2 +
                      (y[:, None] - cy * height)**2 <= (radius * width)**2)
            rows = (y >= top * height) & (y < bottom * height)
            self._masks[width, height] = inside & rows[:, None]
        return self._masks[width, height]

    def __repr__(self):
        return "SkyRegion(circle=%s, rows=%s)" % (self.circle, self.rows)


class SunBlockerDetector:
    """Flag tubelets that show the sun blocker.

    Blocker pixels are found on the stored frames, before augmentation. The
    background of each pixel is a grayscale closing of the frame (a ``kernel``
    x ``kernel`` max filter, then a min filter). It fills in dark features
    too narrow to hold a kernel-sized square, such as the disc and the arm,
    with the brightness of the sky around them, and leaves wider dark areas
    such as deep-blue sky as they are. A pixel counts as blocker when it is
    at least ``contrast`` darker than its background, as a fraction of the
    background brightness. The caller crops the per-frame masks together
    with the frames and turns them into tubelet flags with ``tubelets``.

    Args:
        num_frames: frames per clip (T).
        tubelet_size: frames per token (temporal patch size).
        patch_size: spatial patch size, int or (ph, pw).
        contrast: a pixel counts when it is at least this fraction darker
            than its background.
        kernel: odd size of the closing in stored-frame pixels. Dark features
            that cannot hold a kernel x kernel square count; the UoH disc is
            about 14 px across in 224x224 frames.
        min_pixels: a tubelet is flagged when it holds at least this many
            blocker pixels.
        sky_region: a SkyRegion; pixels outside it never count. None counts
            every pixel.
        floor: background brightness in [0, 1] below which darkening is
            measured against ``floor`` instead, so that small differences in
            dark areas such as trees do not count.
    """

    def __init__(self,
                 num_frames,
                 tubelet_size,
                 patch_size,
                 contrast=0.5,
                 kernel=15,
                 min_pixels=1,
                 sky_region=None,
                 floor=0.2):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if num_frames % tubelet_size != 0:
            raise ValueError(
                f"num_frames={num_frames} is not a multiple of "
                f"tubelet_size={tubelet_size}")
        if not 0 < contrast < 1:
            raise ValueError("contrast must lie in (0, 1)")
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("kernel must be an odd number of at least 3")
        if min_pixels < 1:
            raise ValueError("min_pixels must be at least 1")
        if not 0 < floor <= 1:
            raise ValueError("floor must lie in (0, 1]")
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.contrast = float(contrast)
        self.kernel = int(kernel)
        self.min_pixels = int(min_pixels)
        self.sky_region = sky_region
        self.floor = float(floor)
        self._square = np.ones((self.kernel, self.kernel), np.uint8)

    def darkness(self, frames):
        """How much darker each pixel is than its background, in [0, 1].

        Args:
            frames: uint8 array ``[T, H, W, 3]`` (or ``[T, H, W]`` gray).
        Returns:
            float32 array ``[T, H, W]``.
        """
        frames = np.asarray(frames)
        if frames.ndim == 4:
            gray = frames.mean(axis=-1, dtype=np.float32)
        else:
            gray = frames.astype(np.float32)
        gray /= 255.
        # OpenCV ignores pixels beyond the border for both passes.
        background = np.stack([
            cv2.morphologyEx(frame, cv2.MORPH_CLOSE, self._square)
            for frame in gray
        ])
        return (background - gray) / np.maximum(background, self.floor)

    def dark_pixels(self, frames):
        """Blocker pixels of stored frames, bool array ``[T, H, W]``."""
        dark = self.darkness(frames) > self.contrast
        if self.sky_region is not None:
            _, h, w = dark.shape
            dark &= self.sky_region.mask(w, h)
        return dark

    def tubelets(self, dark):
        """Tubelet mask ``[T // tubelet_size, H // ph, W // pw]``.

        True marks a tubelet that holds at least ``min_pixels`` of the
        ``[T, H, W]`` blocker pixels ``dark``.
        """
        dark = np.asarray(dark, dtype=bool)
        t, h, w = dark.shape
        ph, pw = self.patch_size
        if h % ph or w % pw:
            raise ValueError(
                f"Frame size {h}x{w} is not divisible by patch size {ph}x{pw}")
        dark = dark.reshape(t // self.tubelet_size, self.tubelet_size,
                            h // ph, ph, w // pw, pw)
        counts = dark.sum(axis=(1, 3, 5))
        return counts >= self.min_pixels

    def __repr__(self):
        return (
            "SunBlockerDetector(contrast=%.2f, kernel=%d, min_pixels=%d, "
            "tubelet_size=%d, patch_size=%s, sky_region=%s)" %
            (self.contrast, self.kernel, self.min_pixels, self.tubelet_size,
             self.patch_size, self.sky_region))


def summarize_flags(flags):
    """Return ``(n_flagged, n_total)`` for a boolean array of clip flags."""
    flags = np.asarray(flags, dtype=bool)
    return int(flags.sum()), int(flags.size)
