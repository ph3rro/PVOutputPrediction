"""UoH frames for checking the sun-blocker detector and calibrating the camera.

Every loader returns 224x224 RGB uint8 frames laid out like the UoH LMDB
(build_uoh_lmdb.py keeps x in [24, 630) of the 640x480 camera frame, pads
black rows to make it square and resizes with INTER_AREA) together with
their UTC times as Unix seconds.
"""
import base64
import calendar
import datetime as dt
import io
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence

REPO = Path(__file__).resolve().parents[4]
NOTEBOOK = REPO / 'preprocessing' / 'video_only_data_preprocess.ipynb'
JUNE_GIF = REPO / 'figures' / 'uoh_2016-06-15.gif'
S = 224
X1, X2 = 24, 630


def unix(text):
    """'YYYY-MM-DD HH:MM:SS' (UTC) -> Unix seconds."""
    return calendar.timegm(
        dt.datetime.strptime(text, '%Y-%m-%d %H:%M:%S').timetuple())


def to_stored(camera_rgb):
    """640x480 camera frame -> 224x224 stored frame."""
    side = X2 - X1
    pad = (side - 480) // 2
    padded = cv2.copyMakeBorder(camera_rgb[:, X1:X2], pad, side - 480 - pad,
                                0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.resize(padded, (S, S), interpolation=cv2.INTER_AREA)


def _notebook_png(cell):
    nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    for output in nb['cells'][cell].get('outputs', []):
        if 'image/png' in output.get('data', {}):
            png = base64.b64decode(output['data']['image/png'])
            return np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))
    raise RuntimeError(f"No image in notebook cell {cell}")


# Overlay times. The 2025 camera clock reads an hour behind UTC: the sun lands
# within 1 px of the camera model only at 12:55:58.
NOTEBOOK_TIMES = {
    '2016-05-04': '2016-05-04 12:32:45',
    '2017-01-17': '2017-01-17 08:10:56',
    '2018-03-16': '2018-03-16 12:21:42',
    '2020-03-07': '2020-03-07 11:17:46',
    '2025-04-30': '2025-04-30 11:55:58',
}


def notebook_frames():
    """The five camera7 frames shown in video_only_data_preprocess.ipynb.

    Returns ``{date: (frame, unix_time)}``. The crop boxes below match the
    figures saved in the notebook and break if those cells are re-run.
    """
    frames = {}
    # Cell 17: four camera frames down the left column of one figure.
    fig = _notebook_png(17)
    for date, (y0, y1) in {'2017-01-17': (81, 443), '2018-03-16': (570, 931),
                           '2020-03-07': (1058, 1420),
                           '2025-04-30': (1547, 1908)}.items():
        camera = cv2.resize(fig[y0:y1, 10:488], (640, 480),
                            interpolation=cv2.INTER_AREA)
        frames[date] = to_stored(camera)
    # Cell 8: one frame with a red circle and a lime box drawn on it.
    img = _notebook_png(8)[32:497, 10:630].copy()
    r, g, b = (img[..., i].astype(int) for i in range(3))
    overlay = (((r > 150) & (g < 90) & (b < 90)) |
               ((g > 150) & (r < 120) & (b < 120))).astype(np.uint8)
    img = cv2.inpaint(img, cv2.dilate(overlay, np.ones((3, 3), np.uint8)), 3,
                      cv2.INPAINT_TELEA)
    frames['2016-05-04'] = to_stored(
        cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA))
    return {d: (frames[d], unix(NOTEBOOK_TIMES[d])) for d in sorted(frames)}


# Sun-blocker disc (centre x, y, radius) and arm polyline in stored-frame
# pixels, marked by hand on 6x zoomed crops of the notebook frames.
BLOCKER = {
    '2016-05-04': ((111, 148.3, 6.0),
                   [(117.3, 147), (129.3, 145.7), (144, 143.7), (160, 139.8),
                    (176, 135.7), (189.3, 131.5), (202.7, 127.3),
                    (212.7, 122.7)]),
    '2018-03-16': ((101.5, 164.5, 6.8),
                   [(107.5, 166.7), (128, 167.7), (144, 167.3), (160, 166.3),
                    (176, 163), (192, 157.7), (200, 153), (204.2, 149.3)]),
    '2020-03-07': ((92, 177, 6.7),
                   [(99, 178.3), (112.3, 179.5), (120.7, 180.3),
                    (135.7, 181.5), (150.7, 181.7), (160, 180.7),
                    (170.7, 179), (180.7, 176.7), (187.3, 174.7),
                    (191.5, 173.3)]),
}


def blocker_masks(date, frame):
    """Bool masks (disc, arm) of the hand-marked blocker; empty if none.

    The arm is snapped to the darkest pixel within 3 px of the polyline in
    each column and drawn 3 px tall.
    """
    disc = np.zeros((S, S), bool)
    arm = np.zeros((S, S), bool)
    if date not in BLOCKER:
        return disc, arm
    (cx, cy, r), poly = BLOCKER[date]
    gray = frame.astype(np.float32).mean(-1)
    yy, xx = np.mgrid[0:S, 0:S] + 0.5
    disc = (xx - cx)**2 + (yy - cy)**2 <= (r + 0.5)**2
    px, py = np.array(poly).T
    for x in range(int(np.ceil(px[0])), int(np.floor(px[-1])) + 1):
        yc = np.interp(x + 0.5, px, py)
        lo = int(round(yc - 3))
        yd = lo + int(np.argmin(gray[lo:int(round(yc + 3)) + 1, x]))
        arm[yd - 1:yd + 2, x] = True
    return disc, arm & ~disc


# Overlay times read off 12 frames of the June GIF; frames in between are
# interpolated (about 59 s apart).
JUNE_ANCHORS = {
    0: '08:44:33', 15: '08:59:20', 30: '09:14:08', 45: '09:28:56',
    60: '09:43:43', 75: '09:58:31', 90: '10:13:22', 105: '10:28:12',
    120: '10:43:00', 135: '10:57:49', 150: '11:12:15', 166: '11:28:09'
}


def gif_frames(path=JUNE_GIF):
    """Frames of the June 15, 2016 timelapse GIF (400x300) and their times."""
    gif = Image.open(path)
    frames = np.stack([
        to_stored(cv2.resize(np.asarray(f.convert('RGB')), (640, 480),
                             interpolation=cv2.INTER_CUBIC))
        for f in ImageSequence.Iterator(gif)
    ])
    idx = np.array(sorted(JUNE_ANCHORS))
    times = np.array([unix('2016-06-15 ' + JUNE_ANCHORS[i]) for i in idx])
    return frames, np.interp(np.arange(len(frames)), idx, times)


def lmdb_day(lmdb_path, stem):
    """All frames of one video in a minute-aligned UoH LMDB, with times.

    ``stem`` is the video stem, e.g. ``camera7_2016-06-15``. The times are
    the UTC minutes the frames were picked for (within 30 s of the overlay).
    """
    import lmdb
    env = lmdb.open(str(lmdb_path), subdir=True, readonly=True, lock=False,
                    readahead=False)
    with env.begin(buffers=True) as txn:
        times = pickle.loads(bytes(txn.get(b'__timestamps__')))[stem]
        frames = [
            np.asarray(
                Image.open(io.BytesIO(bytes(txn.get(
                    f"{stem}@{t}".encode('utf-8'))))).convert('RGB'))
            for t in times
        ]
    env.close()
    return np.stack(frames), np.asarray(times, dtype=np.float64)
