# Sun-blocker tools

Scripts for checking the `--sun_blocker_masking` detector (`dataset/sun_blocker.py`) and for predicting where the UoH sun blocker is from the date and time. Run them from `models/VideoMAEv2`.

| File | What it does |
|---|---|
| `test_sun_blocker.py` | Checks for the detector, the mask cropping, the solar position and the camera model |
| `evaluate_detector.py` | Scores `--sun_blocker_contrast` values on the hand-marked frames and on a whole day |
| `render_figures.py` | Draws the pictures: frames next to their detections, and predicted sun positions |
| `calibrate_camera.py` | Tracks the blocker disc through a day, fits the camera model, checks it and looks for clock offsets |
| `frames.py` | Loads frames: five from the preprocessing notebook (with hand-marked blocker), the June 15, 2016 GIF, or any day of the minute-aligned LMDB |
| `sunpos.py` | Sun azimuth and zenith from UTC time at Bayfordbury Observatory (51.7753 N, 0.0950 W) |
| `camera.py` | Fisheye camera model (sun direction to pixel in the 224x224 stored frames) and the June 2016 fit |

```bash
cd /home/ubuntu/PVOutputPrediction/models/VideoMAEv2
LMDB=/home/ubuntu/PVOutputPrediction/preprocessing/data/uoh_lmdb_minute_224/frames.lmdb

python tools/sun_blocker/test_sun_blocker.py
python tools/sun_blocker/evaluate_detector.py --lmdb_path $LMDB --stem camera7_2016-06-15
python tools/sun_blocker/render_figures.py marked day predicted --lmdb_path $LMDB --stem camera7_2016-06-15 --out_dir sun_blocker_vis
python tools/sun_blocker/calibrate_camera.py --lmdb_path $LMDB --stem camera7_2016-06-15 --window 08:00 14:00 --no_fit
```

On the Windows machine, use `--gif ../../figures/uoh_2016-06-15.gif` instead of `--lmdb_path`/`--stem`. That GIF is not in git.

## Findings so far (October 2026)

**Detector.** The old fixed gray threshold (60/255) caught 16 of 33 hand-marked blocker patches and wrongly flagged 375 others on the five notebook frames. The thin arm blends to gray 75-90 at 224 px, and sun glare can light the disc to gray 82. The current detector flags pixels at least `--sun_blocker_contrast` darker than a 15 px grayscale closing of the frame, inside a circle 5 px inside the camera's dark rim.

| Contrast | Blocker patches found (notebook frames) | Other patches flagged (notebook frames) | June 15, 2016: patches flagged per frame (of 196) |
|---|---|---|---|
| 0.35 | 30/33 | 65 | 90 |
| 0.50 (default) | 26/33 | 30 | 42 |
| 0.60 | 20/33 | 19 | 25 |

The weak spot is broken cumulus: deep-blue gaps between bright clouds look as dark as the blocker. Color does not separate them. The blocker's pixels are about as blue as the sky (the 1-2 px arm takes the color of the sky behind it), so skipping blue pixels keeps only 9 of 33 blocker patches. Skipping pixels bluer than their surroundings by more than 0.10 keeps 26/33 and cuts June flags from 42 to 34; this is not implemented.

**Position from date and time.** The blocker disc sits on the sun, so its position follows from the timestamp. `sunpos.py` matches pvlib to 0.02 degrees. A camera model fitted on the 167 frames of the June 15, 2016 GIF puts the sun within 0.4 px (median) of the disc, and lands 3 px and 1 px from the hand-marked discs of 2016-05-04 and 2018-03-16. Two frames disagree:

- 2020-03-07 11:17:46: 22 px off, not explained. No clean clock offset fits (the best, +52 min, leaves 10 px), and one frame cannot tell whether the camera moved or the blocker was off the sun.
- 2025-04-30 11:55:58 (no blocker; the sun itself was checked): 21 px off as labelled, 1 px if the true time is one hour later. That frame's "UTC" overlay is an hour behind, so the timestamps cannot be fully trusted.

LMDB timestamps are the UTC minutes chosen from the decoded overlay (`preprocessing/uoh_timestamp.py`), within 30 s of the overlay time.

## Next steps

1. Calibrate on clear days spread over 2016 to June 2020 from the LMDB, check each day for a clock offset (`calibrate_camera.py` prints the best one), and find what changed by 2020.
2. Model the arm. It follows the sun too: it hangs down toward the bottom edge on June mornings and points right around midday.
3. Use the predicted disc and arm as a corridor a few pixels wide and keep only detector pixels inside it, so cloud gaps elsewhere never count. This needs each clip's timestamps passed from `LMDBVideoMAE` to the transform.

The five notebook frames are rebuilt from figures saved in `preprocessing/video_only_data_preprocess.ipynb`. The crop boxes in `frames.py` match those saved outputs and break if cells 8 or 17 are re-run.
