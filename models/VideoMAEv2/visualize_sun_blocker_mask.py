#!/usr/bin/env python
"""Visualise what --sun_blocker_masking would mask on real LMDB clips.

Runs the exact training augmentation and sun-blocker detector on a few clips
and writes one PNG per clip to --out_dir. Each PNG has one row per shown frame
and four panels per row:

  1. augmented frame as the model sees it
  2. dark pixels (below --sun_blocker_threshold) tinted red
  3. tubelets flagged as sun blocker (excluded from the loss) shaded magenta
  4. encoder input: patches hidden from the encoder are blacked out; every
     flagged patch is always among them

Use it to tune --sun_blocker_threshold / --sun_blocker_min_pixels before a
pretraining run, e.g.

  python visualize_sun_blocker_mask.py \\
      --lmdb_path /data/uoh_lmdb_minute_224/frames.lmdb \\
      --clip_stride_minutes 2 --stem_contains 2018-03-16 --num_clips 4
"""
import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from dataset.pretrain_datasets import DataAugmentationForVideoMAEv2, LMDBVideoMAE
from dataset.sun_blocker import date_from_stem, parse_date


def get_args():
    parser = argparse.ArgumentParser(
        'Visualise sun-blocker masking', add_help=True)
    parser.add_argument('--lmdb_path', required=True, type=str)
    parser.add_argument('--out_dir', default='sun_blocker_vis', type=str)
    parser.add_argument('--num_clips', default=6, type=int)
    parser.add_argument(
        '--indices',
        default=None,
        type=int,
        nargs='+',
        help='Explicit clip indices; overrides --num_clips/--stem_contains')
    parser.add_argument(
        '--stem_contains',
        default=None,
        type=str,
        help='Only consider clips whose video stem contains this text, '
        'e.g. a date such as 2018-03-16')
    parser.add_argument(
        '--frames_to_show',
        default=3,
        type=int,
        help='How many frames of the clip to draw (evenly spaced)')
    parser.add_argument('--seed', default=0, type=int)
    # Same names and defaults as run_mae_pretraining.py.
    parser.add_argument('--clip_stride_minutes', default=None, type=int)
    parser.add_argument('--dataset_limit', default=None, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--sampling_rate', default=4, type=int)
    parser.add_argument('--tubelet_size', default=2, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--mask_ratio', default=0.9, type=float)
    parser.add_argument('--sun_blocker_until', default='2020-07-01', type=str)
    parser.add_argument('--sun_blocker_threshold', default=60, type=float)
    parser.add_argument('--sun_blocker_min_pixels', default=1, type=int)
    return parser.parse_args()


def build(args):
    aug_args = SimpleNamespace(
        input_size=args.input_size,
        mask_type='tube',
        mask_ratio=args.mask_ratio,
        decoder_mask_ratio=0.0,
        decoder_mask_type='run_cell',
        window_size=(args.num_frames // args.tubelet_size,
                     args.input_size // args.patch_size,
                     args.input_size // args.patch_size),
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        patch_size=(args.patch_size, args.patch_size),
        sun_blocker_masking=True,
        sun_blocker_threshold=args.sun_blocker_threshold,
        sun_blocker_min_pixels=args.sun_blocker_min_pixels,
    )
    transform = DataAugmentationForVideoMAEv2(aug_args)
    dataset = LMDBVideoMAE(
        lmdb_path=args.lmdb_path,
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        num_sample=1,
        key_limit=args.dataset_limit,
        clip_stride_minutes=args.clip_stride_minutes,
        sun_blocker_until=parse_date(args.sun_blocker_until))
    return transform, dataset


def clip_stem(dataset, index):
    clip = dataset.clips[index]
    key = clip[0] if isinstance(clip, tuple) else clip
    return key.decode('utf-8')


def choose_indices(args, dataset):
    if args.indices:
        return list(args.indices)
    candidates = range(len(dataset))
    if args.stem_contains:
        candidates = [
            i for i in candidates
            if args.stem_contains in clip_stem(dataset, i)
        ]
        if not candidates:
            raise SystemExit(
                f"No clip stem contains {args.stem_contains!r}")
    candidates = list(candidates)
    n = min(args.num_clips, len(candidates))
    picks = np.linspace(0, len(candidates) - 1, num=n, dtype=int)
    return [candidates[i] for i in picks]


def to_uint8_frames(process_data, transform):
    """[3, T, H, W] normalised -> [T, H, W, 3] uint8."""
    mean = torch.tensor(transform.input_mean).view(3, 1, 1, 1)
    std = torch.tensor(transform.input_std).view(3, 1, 1, 1)
    frames = (process_data * std + mean).clamp(0, 1)
    return (frames.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)


def tint(frame, mask, color, alpha=0.6):
    out = frame.astype(np.float32)
    color = np.asarray(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * color
    return out.round().astype(np.uint8)


def upsample_grid(grid, patch):
    """[h, w] bool patch grid -> [h*patch, w*patch] bool pixel mask."""
    return np.repeat(np.repeat(grid, patch, axis=0), patch, axis=1)


def render_clip(process_data, encoder_mask, exclude_mask, transform, args):
    frames = to_uint8_frames(process_data, transform)  # [T, H, W, 3]
    t_total = frames.shape[0]
    flat = process_data.transpose(0, 1).reshape(t_total * 3, *frames.shape[1:3])
    dark = transform.sun_blocker_detector.dark_pixels(flat).numpy()  # [T,H,W]
    grid = args.input_size // args.patch_size
    exclude = np.asarray(exclude_mask, dtype=bool).reshape(-1, grid, grid)
    encoder = np.asarray(encoder_mask, dtype=bool).reshape(-1, grid, grid)

    shown = np.linspace(0, t_total - 1, num=min(args.frames_to_show, t_total),
                        dtype=int)
    rows = []
    for t in shown:
        slot = t // args.tubelet_size
        frame = frames[t]
        panel_dark = tint(frame, dark[t], (255, 0, 0))
        panel_excl = tint(frame, upsample_grid(exclude[slot], args.patch_size),
                          (255, 0, 255), alpha=0.5)
        panel_enc = frame.copy()
        panel_enc[upsample_grid(encoder[slot], args.patch_size)] = 0
        rows.append(np.concatenate(
            [frame, panel_dark, panel_excl, panel_enc], axis=1))
    return np.concatenate(rows, axis=0), exclude, encoder


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    transform, dataset = build(args)
    print(transform)
    for index in choose_indices(args, dataset):
        stem = clip_stem(dataset, index)
        outputs = dataset[index]
        process_data, encoder_mask = outputs[0], outputs[1]
        exclude_mask = outputs[3]
        canvas, exclude, encoder = render_clip(
            process_data, encoder_mask, exclude_mask, transform, args)
        flagged = bool(dataset.clip_sun_blocker[index])
        per_slot = exclude.reshape(exclude.shape[0], -1).sum(axis=1)
        hidden_ok = bool(np.all(encoder[exclude]))
        name = f"clip{index:07d}_{stem}.png"
        Image.fromarray(canvas).save(os.path.join(args.out_dir, name))
        print(
            f"{name}: date={date_from_stem(stem)} flagged={flagged} "
            f"excluded tubelets per time slot={per_slot.tolist()} "
            f"({100 * exclude.mean():.1f}% of tubelets) "
            f"all excluded patches hidden from encoder={hidden_ok}")
    print(f"Wrote images to {os.path.abspath(args.out_dir)}")


if __name__ == '__main__':
    main()
