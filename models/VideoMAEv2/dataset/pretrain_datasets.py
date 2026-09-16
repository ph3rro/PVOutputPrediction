import io
import os
import pickle
import random

import lmdb
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .loader import get_image_loader, get_video_loader
from .masking_generator import (
    RunningCellMaskingGenerator,
    TubeMaskingGenerator,
)
from .sun_blocker import (
    SunBlockerDetector,
    clip_has_sun_blocker,
    date_from_stem,
)
from .transforms import (
    GroupMultiScaleCrop,
    GroupNormalize,
    Stack,
    ToTorchFormatTensor,
)


class DataAugmentationForVideoMAEv2(object):

    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        div = True
        roll = False
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size,
                                                      [1, .875, .75, .66])
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=roll),
            ToTorchFormatTensor(div=div),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.encoder_mask_map_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio)
        else:
            raise NotImplementedError(
                'Unsupported encoder masking strategy type.')
        if args.decoder_mask_ratio > 0.:
            if args.decoder_mask_type == 'run_cell':
                self.decoder_mask_map_generator = RunningCellMaskingGenerator(
                    args.window_size, args.decoder_mask_ratio)
            else:
                raise NotImplementedError(
                    'Unsupported decoder masking strategy type.')
        # Optional sun-blocker masking (see dataset/sun_blocker.py). When it
        # is enabled, __call__ returns a fourth item: a bool map of tubelets
        # to exclude from the reconstruction loss.
        self.sun_blocker_detector = None
        if getattr(args, 'sun_blocker_masking', False):
            self.sun_blocker_detector = SunBlockerDetector(
                num_frames=args.num_frames,
                tubelet_size=args.tubelet_size,
                patch_size=args.patch_size,
                threshold=args.sun_blocker_threshold,
                min_pixels=args.sun_blocker_min_pixels,
                mean=self.input_mean,
                std=self.input_std)

    def __call__(self, images, sun_blocker=False):
        """Augment a clip and sample its masks.

        Args:
            images: ``(list_of_PIL_frames, label)``.
            sun_blocker: run the sun-blocker detector on this clip. Only
                meaningful when the detector is enabled.
        Returns:
            ``(process_data, encoder_mask, decoder_mask)`` and, when the
            detector is enabled, a fourth ``loss_exclude`` bool map with the
            same layout as the encoder mask ([frames // tubelet_size,
            patches per frame]).
        """
        process_data, _ = self.transform(images)
        sun_blocker_map = None
        if self.sun_blocker_detector is not None and sun_blocker:
            tubelet_mask = self.sun_blocker_detector(process_data)
            sun_blocker_map = tubelet_mask.reshape(
                tubelet_mask.shape[0], -1).numpy()
        if sun_blocker_map is None:
            encoder_mask_map = self.encoder_mask_map_generator()
        else:
            # Tube masking keeps one spatial pattern for the whole clip, so
            # any patch the blocker touches at any time is hidden throughout.
            encoder_mask_map = self.encoder_mask_map_generator(
                forced_mask=sun_blocker_map.any(axis=0))
        if hasattr(self, 'decoder_mask_map_generator'):
            decoder_mask_map = self.decoder_mask_map_generator()
        else:
            decoder_mask_map = 1 - encoder_mask_map
        if self.sun_blocker_detector is None:
            return process_data, encoder_mask_map, decoder_mask_map
        if sun_blocker_map is None:
            sun_blocker_map = np.zeros(encoder_mask_map.shape, dtype=bool)
        return (process_data, encoder_mask_map, decoder_mask_map,
                sun_blocker_map)

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAEv2,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Encoder Masking Generator = %s,\n" % str(
            self.encoder_mask_map_generator)
        if hasattr(self, 'decoder_mask_map_generator'):
            repr += "  Decoder Masking Generator = %s,\n" % str(
                self.decoder_mask_map_generator)
        else:
            repr += "  Do not use decoder masking,\n"
        if self.sun_blocker_detector is not None:
            repr += "  Sun-blocker masking = %s,\n" % str(
                self.sun_blocker_detector)
        repr += ")"
        return repr


class HybridVideoMAE(torch.utils.data.Dataset):
    """Load your own videomae pretraining dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are four items in each line:
        (1) video path; (2) start_idx, (3) total frames and (4) video label.
        for pre-train video data
            total frames < 0, start_idx and video label meaningless
        for pre-train rawframe data
            video label meaningless
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default 'img_{:05}.jpg'.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    transform : function, default None.
        A function that takes data and label and transforms them.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    num_sample : int, default 1.
        Number of sampled views for Repeated Augmentation.
    """

    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_{:05}.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 lazy_init=False,
                 num_sample=1):

        super(HybridVideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_ext = video_ext
        self.transform = transform
        self.lazy_init = lazy_init
        self.num_sample = num_sample

        # NOTE:
        # for hybrid train
        # different frame naming formats are used for different datasets
        # should MODIFY the fname_tmpl to your own situation
        self.ava_fname_tmpl = 'image_{:06}.jpg'
        self.ssv2_fname_tmpl = 'img_{:05}.jpg'

        # NOTE:
        # we set sampling_rate = 2 for ssv2
        # thus being consistent with the fine-tuning stage
        # Note that the ssv2 we use is decoded to frames at 12 fps;
        # if decoded at 24 fps, the sample interval should be 4.
        self.orig_new_step = new_step
        self.orig_skip_length = self.skip_length
        
        self.video_loader = get_video_loader()
        self.image_loader = get_image_loader()

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise (
                    RuntimeError("Found 0 video clips in subfolders of: " +
                                 root + "\n"
                                 "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        try:
            video_name, start_idx, total_frame = self.clips[index]
            self.skip_length = self.orig_skip_length
            self.new_step = self.orig_new_step
            
            if total_frame < 0:
                decord_vr = self.video_loader(video_name)
                duration = len(decord_vr)

                segment_indices, skip_offsets = self._sample_train_indices(
                    duration)
                frame_id_list = self.get_frame_id_list(duration,
                                                       segment_indices,
                                                       skip_offsets)
                video_data = decord_vr.get_batch(frame_id_list).asnumpy()
                images = [
                    Image.fromarray(video_data[vid, :, :, :]).convert('RGB')
                    for vid, _ in enumerate(frame_id_list)
                ]

            else:
                # ssv2 & ava & other rawframe dataset
                if 'SomethingV2' in video_name:
                    self.new_step = 2
                    self.skip_length = self.new_length * self.new_step
                    fname_tmpl = self.ssv2_fname_tmpl
                elif 'AVA2.2' in video_name:
                    fname_tmpl = self.ava_fname_tmpl
                else:
                    fname_tmpl = self.name_pattern

                segment_indices, skip_offsets = self._sample_train_indices(
                    total_frame)
                frame_id_list = self.get_frame_id_list(total_frame,
                                                       segment_indices,
                                                       skip_offsets)

                images = []
                for idx in frame_id_list:
                    frame_fname = os.path.join(
                        video_name, fname_tmpl.format(idx + start_idx))
                    img = self.image_loader(frame_fname)
                    img = Image.fromarray(img)
                    images.append(img)

        except Exception as e:
            print("Failed to load video from {} with error {}".format(
                video_name, e))
            index = random.randint(0, len(self.clips) - 1)
            return self.__getitem__(index)

        if self.num_sample > 1:
            process_data_list = []
            encoder_mask_list = []
            decoder_mask_list = []
            for _ in range(self.num_sample):
                process_data, encoder_mask, decoder_mask = self.transform(
                    (images, None))
                process_data = process_data.view(
                    (self.new_length, 3) + process_data.size()[-2:]).transpose(
                        0, 1)
                process_data_list.append(process_data)
                encoder_mask_list.append(encoder_mask)
                decoder_mask_list.append(decoder_mask)
            return process_data_list, encoder_mask_list, decoder_mask_list
        else:
            process_data, encoder_mask, decoder_mask = self.transform(
                (images, None))
            # T*C,H,W -> T,C,H,W -> C,T,H,W
            process_data = process_data.view(
                (self.new_length, 3) + process_data.size()[-2:]).transpose(
                    0, 1)
            return process_data, encoder_mask, decoder_mask

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, root, setting):
        if not os.path.exists(setting):
            raise (RuntimeError(
                "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. "
                % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise (RuntimeError(
                        'Video input format is not correct, missing one or more element. %s'
                        % line))
                clip_path = os.path.join(root, line_info[0])
                start_idx = int(line_info[1])
                total_frame = int(line_info[2])
                item = (clip_path, start_idx, total_frame)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length +
                            1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(
                    num_frames - self.skip_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def get_frame_id_list(self, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return frame_id_list


class VideoMAE(torch.utils.data.Dataset):
    """Load your own videomae pretraining dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are four items in each line:
        (1) video path; (2) start_idx, (3) total frames and (4) video label.
        for pre-train video data
            total frames < 0, start_idx and video label meaningless
        for pre-train rawframe data
            video label meaningless
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default 'img_{:05}.jpg'.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    transform : function, default None.
        A function that takes data and label and transforms them.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    num_sample : int, default 1.
        Number of sampled views for Repeated Augmentation.
    """

    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_{:05}.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 lazy_init=False,
                 num_sample=1):

        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_ext = video_ext
        self.transform = transform
        self.lazy_init = lazy_init
        self.num_sample = num_sample

        self.video_loader = get_video_loader()
        self.image_loader = get_image_loader()

        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise (
                    RuntimeError("Found 0 video clips in subfolders of: " +
                                 root + "\n"
                                 "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        try:
            video_name, start_idx, total_frame = self.clips[index]
            if total_frame < 0:  # load video
                decord_vr = self.video_loader(video_name)
                duration = len(decord_vr)

                segment_indices, skip_offsets = self._sample_train_indices(
                    duration)
                frame_id_list = self.get_frame_id_list(duration,
                                                       segment_indices,
                                                       skip_offsets)
                video_data = decord_vr.get_batch(frame_id_list).asnumpy()
                images = [
                    Image.fromarray(video_data[vid, :, :, :]).convert('RGB')
                    for vid, _ in enumerate(frame_id_list)
                ]
            else:  # load frames
                segment_indices, skip_offsets = self._sample_train_indices(
                    total_frame)
                frame_id_list = self.get_frame_id_list(total_frame,
                                                       segment_indices,
                                                       skip_offsets)

                images = []
                for idx in frame_id_list:
                    frame_fname = os.path.join(
                        video_name, self.name_pattern.format(idx + start_idx))
                    img = self.image_loader(frame_fname)
                    img = Image.fromarray(img)
                    images.append(img)

        except Exception as e:
            print("Failed to load video from {} with error {}".format(
                video_name, e))
            index = random.randint(0, len(self.clips) - 1)
            return self.__getitem__(index)

        if self.num_sample > 1:
            process_data_list = []
            encoder_mask_list = []
            decoder_mask_list = []
            for _ in range(self.num_sample):
                process_data, encoder_mask, decoder_mask = self.transform(
                    (images, None))
                process_data = process_data.view(
                    (self.new_length, 3) + process_data.size()[-2:]).transpose(
                        0, 1)
                process_data_list.append(process_data)
                encoder_mask_list.append(encoder_mask)
                decoder_mask_list.append(decoder_mask)
            return process_data_list, encoder_mask_list, decoder_mask_list
        else:
            process_data, encoder_mask, decoder_mask = self.transform(
                (images, None))
            # T*C,H,W -> T,C,H,W -> C,T,H,W
            process_data = process_data.view(
                (self.new_length, 3) + process_data.size()[-2:]).transpose(
                    0, 1)
            return process_data, encoder_mask, decoder_mask

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, root, setting):
        if not os.path.exists(setting):
            raise (RuntimeError(
                "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. "
                % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, start_idx, total_frames
                if len(line_info) < 3:
                    raise (RuntimeError(
                        'Video input format is not correct, missing one or more element. %s'
                        % line))
                clip_path = os.path.join(root, line_info[0])
                start_idx = int(line_info[1])
                total_frame = int(line_info[2])
                item = (clip_path, start_idx, total_frame)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length +
                            1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration)
            offsets = offsets + np.random.randint(
                average_duration, size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(
                np.random.randint(
                    num_frames - self.skip_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments, ))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def get_frame_id_list(self, duration, indices, skip_offsets):
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        return frame_id_list


class LMDBVideoMAE(VideoMAE):
    """VideoMAE pretraining dataset backed by one LMDB value per video.

    Each non-metadata key in the database identifies a video. Its value is a
    pickled list of encoded image bytes, as written by build_uoh_lmdb.py.
    LMDB environments are opened lazily in each DataLoader worker because an
    environment must not be shared across forked processes.

    When ``sun_blocker_until`` (a ``datetime.date``) is given, clips whose
    video stem is dated on or before it are flagged and the transform is
    called with ``sun_blocker=True`` for them, so it can mask the UoH sun
    blocker (see dataset/sun_blocker.py). Videos without a parseable
    ``YYYY-MM-DD`` in their stem are never flagged.
    """

    def __init__(self,
                 lmdb_path,
                 new_length=16,
                 new_step=4,
                 transform=None,
                 temporal_jitter=False,
                 num_sample=1,
                 key_limit=None,
                 clip_stride_minutes=None,
                 sun_blocker_until=None):
        super().__init__(
            root='',
            setting='',
            train=True,
            test_mode=False,
            is_color=True,
            modality='rgb',
            num_segments=1,
            num_crop=1,
            new_length=new_length,
            new_step=new_step,
            transform=transform,
            temporal_jitter=temporal_jitter,
            lazy_init=True,
            num_sample=num_sample)

        self.lmdb_path = os.path.abspath(lmdb_path)
        self.clip_stride_minutes = clip_stride_minutes
        if not os.path.isdir(self.lmdb_path):
            raise FileNotFoundError(
                f"LMDB directory does not exist: {self.lmdb_path}")

        # Read keys without touching the large values. This also works for a
        # partially built database that does not yet contain __keys__.
        env = self._open_env()
        with env.begin(write=False) as txn:
            stored_keys = txn.get(b'__keys__')
            if stored_keys is not None:
                keys = pickle.loads(stored_keys)
                keys = [
                    key.encode('utf-8') if isinstance(key, str) else key
                    for key in keys
                ]
            else:
                keys = [
                    key for key in txn.cursor().iternext(
                        keys=True, values=False)
                    if not key.startswith(b'__') and b'@' not in key
                ]
            stored_timestamps = txn.get(b'__timestamps__')
            video_timestamps = (
                pickle.loads(stored_timestamps)
                if stored_timestamps is not None else None
            )
        env.close()

        keys = sorted(keys)
        if key_limit is not None:
            keys = keys[:key_limit]

        if clip_stride_minutes is not None:
            if video_timestamps is None:
                raise RuntimeError(
                    "--clip_stride_minutes requires a minute-aligned LMDB")
            self.clips = []
            stride = int(clip_stride_minutes)
            if stride < 1:
                raise ValueError("clip_stride_minutes must be at least 1")
            for key in keys:
                stem = key.decode('utf-8')
                timestamps = video_timestamps.get(stem, [])
                run_start = 0
                for end in range(1, len(timestamps) + 1):
                    run_ended = (
                        end == len(timestamps)
                        or timestamps[end] - timestamps[end - 1] != 60
                    )
                    if not run_ended:
                        continue
                    last_start = end - self.new_length
                    for start in range(
                            run_start, last_start + 1, stride):
                        self.clips.append((key, start))
                    run_start = end
            self.video_timestamps = video_timestamps
        else:
            self.clips = keys
            self.video_timestamps = None

        if not self.clips:
            raise RuntimeError(f"No video entries found in {self.lmdb_path}")

        self.sun_blocker_until = sun_blocker_until
        self.clip_sun_blocker = None
        if sun_blocker_until is not None:
            key_flags = {
                key: clip_has_sun_blocker(
                    key.decode('utf-8'), sun_blocker_until)
                for key in keys
            }
            self.clip_sun_blocker = np.fromiter(
                (key_flags[clip[0] if isinstance(clip, tuple) else clip]
                 for clip in self.clips),
                dtype=bool,
                count=len(self.clips))
            n_undated = sum(
                date_from_stem(key.decode('utf-8')) is None for key in keys)
            print(
                f"Sun-blocker masking: {int(self.clip_sun_blocker.sum())} of "
                f"{len(self.clips)} clips are dated on or before "
                f"{sun_blocker_until} and will be masked"
                + (f" ({n_undated} videos have no parseable date and are "
                   "never masked)" if n_undated else ""))

        self._env = None
        if clip_stride_minutes is None:
            print(
                f"Loaded {len(self.clips)} LMDB videos from {self.lmdb_path}")
        else:
            print(
                f"Loaded {len(self.clips)} minute-aligned clips from "
                f"{len(keys)} LMDB videos at a {clip_stride_minutes}-minute "
                "stride")

    def _open_env(self):
        return lmdb.open(
            self.lmdb_path,
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512)

    def _get_env(self):
        if self._env is None:
            self._env = self._open_env()
        return self._env

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_env'] = None
        return state

    def __del__(self):
        env = getattr(self, '_env', None)
        if env is not None:
            env.close()

    def _load_images(self, index):
        clip = self.clips[index]
        if isinstance(clip, tuple):
            key, start = clip
            stem = key.decode('utf-8')
            timestamps = self.video_timestamps[stem][
                start:start + self.new_length]
            with self._get_env().begin(write=False, buffers=True) as txn:
                encoded_frames = []
                for timestamp in timestamps:
                    frame_key = f"{stem}@{timestamp}".encode('utf-8')
                    payload = txn.get(frame_key)
                    if payload is None:
                        raise KeyError(f"Missing LMDB key: {frame_key!r}")
                    encoded_frames.append(bytes(payload))
            return [
                Image.open(io.BytesIO(frame)).convert('RGB')
                for frame in encoded_frames
            ]

        key = clip
        with self._get_env().begin(write=False, buffers=True) as txn:
            payload = txn.get(key)
            if payload is None:
                raise KeyError(f"Missing LMDB key: {key!r}")
            encoded_frames = pickle.loads(bytes(payload))

        duration = len(encoded_frames)
        if duration < self.skip_length:
            raise RuntimeError(
                f"{key.decode(errors='replace')} has {duration} frames; "
                f"{self.skip_length} are required")

        segment_indices, skip_offsets = self._sample_train_indices(duration)
        frame_ids = self.get_frame_id_list(
            duration, segment_indices, skip_offsets)
        return [
            Image.open(io.BytesIO(encoded_frames[frame_id])).convert('RGB')
            for frame_id in frame_ids
        ]

    def __getitem__(self, index):
        # Retry another video on isolated corruption without recursing forever.
        for _ in range(10):
            try:
                images = self._load_images(index)
                break
            except Exception as exc:
                clip = self.clips[index]
                key = clip[0] if isinstance(clip, tuple) else clip
                key = key.decode(errors='replace')
                print(f"Failed to load LMDB video {key}: {exc}")
                index = random.randrange(len(self.clips))
        else:
            raise RuntimeError("Failed to load 10 LMDB videos in a row")

        # `index` may have been redrawn above, so look the flag up here.
        sun_blocker = (
            bool(self.clip_sun_blocker[index])
            if self.clip_sun_blocker is not None else False)

        if self.num_sample > 1:
            samples = [
                self._transform_clip(images, sun_blocker)
                for _ in range(self.num_sample)
            ]
            # One list per field: (process_data_list, encoder_mask_list,
            # decoder_mask_list[, loss_exclude_list]).
            return tuple(list(field) for field in zip(*samples))

        return self._transform_clip(images, sun_blocker)

    def _transform_clip(self, images, sun_blocker):
        """Augment one clip; returns (process_data, *masks)."""
        if self.clip_sun_blocker is None:
            outputs = self.transform((images, None))
        else:
            outputs = self.transform((images, None), sun_blocker=sun_blocker)
        process_data = outputs[0]
        # T*C,H,W -> T,C,H,W -> C,T,H,W
        process_data = process_data.view(
            (self.new_length, 3) +
            process_data.size()[-2:]).transpose(0, 1)
        return (process_data,) + tuple(outputs[1:])
