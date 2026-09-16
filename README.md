# PV Output Forecasting from Sky Images with VideoMAEv2

Data preprocessing scripts for sky-video-PV pairs and modified VideoMAEv2 for regression and PV output forecasting from 15-minute sky videos, with a forecast horizon of 15 minutes.

# Installation

## Prerequisites

- Python 3.12.3
- NVIDIA GPU with CUDA support (recommended for training)
- CUDA 12.6+ (version compatible with Pytorch)
- Git

## Step 1: Clone the Repository

```bash
git clone https://github.com/ph3rro/PVOutputPrediction
cd PVOutputPrediction
```

## Step 2: Create Virtual Environments

Install Python 3.12.3 and add to PATH.

You will need two separate virtual environments—one for the preprocessing notebooks and one for VideoMAEv2

```bash
python3.12 -m venv preprocessing-env
cd models/VideoMAEv2
python3.12 -m venv VideoMAE-env
```

## Step 3: Install PyTorch with CUDA Support 

Install PyTorch first (on both environments), as it requires specific CUDA versions. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to get the appropriate command for your system.

For CUDA 13.0 (as used in this project):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Step 4: Install Core Dependencies

Install the main project dependencies in two separate virtual environments for preprocessing notebooks and VideoMAE:

### Windows
```bash
preprocessing-env\Scripts\activate
pip install -r requirements.txt
deactivate
cd models/VideoMAEv2
VideoMAE-env\Scripts\activate
pip install -r requirements-MAE.txt
```

### Linux 

```bash
source preprocessing-env/bin/activate
pip install -r requirements.txt
deactivate
cd models/VideoMAEv2
source VideoMAE-env/bin/activate
pip install -r requirements-MAE.txt
```

This will install:
- Core libraries: numpy, pandas, scipy
- Deep learning: scikit-learn, timm, transformers
- Computer vision: opencv-python, pillow
- Data handling: h5py, hdf5plugin
- Utilities: CRPS, tqdm, tensorboard, matplotlib
- Jupyter notebook support

## Step 5: Verify Installation

Check that PyTorch can access your GPU:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

# Training

## Finetuning

### CUDA
Most of the experiments we ran used 2xA100. You can specify which DATA_PATH you want to use. This folder should contain folders named metadata_trainval, metadata_test, videos_trainval, and videos_test. The metadata folders should contain a parquet file with columns of mp4 file name, timestamp, pv log of shape 16, and pv pred, while the videos folders should contain enumerated mp4s. These folders can be generated using the preprocessing code. For finetuning on SKIPP'D, the command would look something like this:

```bash
NUM_GPU=2
DATA_PATH="/home/ubuntu/PVOutputPrediction/preprocessing/data/data_forecast4"
OUTPUT_DIR="/home/ubuntu/PVOutputPrediction/models/VideoMAEv2/checkpoints_residual5"
NUM_WORKERS=8
torchrun --nproc_per_node=${NUM_GPU} run_class_finetuning.py --batch_size=48 --lr=5e-4 --num_workers={NUM_WORKERS} --mixup=0 --cutmix=0 --warmup_epochs=1 --layer_decay=0.9 --dist_eval --weight_decay=0.1 --data_path=${DATA_PATH} --device='cuda' --enable_deepspeed --output_dir=${OUTPUT_DIR} --save_ckpt --save_ckpt_freq=3 --use_residual --log_dir=${OUTPUT_DIR} --clip_grad=1.0
```

For finetuning on Pangaea, since the dataset is larger, we use LMDB to store PNGs. The 16 consecutive frames can then be quickly accessed at train time, and don't have to be bundled together. This avoids an overlap and reduces the storage size by 8x. 

```bash
cd /home/ubuntu/PVOutputPrediction/models/VideoMAEv2
./run_pangaea_lmdb_finetune.sh
```

You can inspect PVOutputPrediction/models/VideoMAEv2/run_pangaea_lmdb_finetune.sh to see how to run it.

### CPU (not recommended/not tested thoroughly)
```bash
DATA_PATH="/home/ubuntu/PVOutputPrediction/preprocessing/data/data_forecast4"
OUTPUT_DIR="/home/ubuntu/PVOutputPrediction/models/VideoMAEv2/checkpoints_residual5"
python run_class_finetuning.py --batch_size=48 --lr=5e-4 --mixup=0 --cutmix=0 --warmup_epochs=1 --layer_decay=0.9 --dist_eval --weight_decay=0.1 --data_path=${DATA_PATH} --device='cuda' --output_dir=${OUTPUT_DIR} --save_ckpt --save_ckpt_freq=3 --use_residual --log_dir="/home/ubuntu/PVOutputPrediction/models/VideoMAEv2/checkpoints_residual5" --clip_grad=1.0
```


## Pretraining

### Linux

To run pretraining on the UoH (University of Hertsfordshire) dataset, using encoder weights from an already pretrained on Kinetics-400, run the following command. 

```bash
cd /home/ubuntu/PVOutputPrediction/models/VideoMAEv2
./run_uoh_pretrain_from_k400.sh
```

### Sun-blocker masking (UoH)

Until 2020-07-01 the UoH camera carried a sun blocker (a black disc on a thin black arm). Add `--sun_blocker_masking` to the pretraining command to keep the model from learning to reconstruct it. For every clip whose LMDB video stem is dated on or before `--sun_blocker_until` (default `2020-07-01`), the augmented frames are thresholded at `--sun_blocker_threshold` (default 60 on a 0-255 gray scale) and every tubelet (2 frames x 16 x 16 pixels) with at least `--sun_blocker_min_pixels` (default 1) dark pixels is

- always placed in the encoder mask (the random tube mask is drawn around it, so the number of visible tokens is unchanged), and
- excluded from the decoder's reconstruction loss.

Note that any other near-black pixels in those clips (the black padding above and below the frame, the corners outside the fisheye disc, the tree line) are masked the same way; clips after the cutoff are untouched. The flag needs `--lmdb_path`, since the clip date is read from the video stem (`camera7_2019-06-20`). Training logs and TensorBoard gain a `sun_blocker_frac` value, the fraction of tubelets excluded per batch. The extra work is a threshold and a per-tubelet count in the DataLoader workers, a few milliseconds per clip, and the GPU step is unchanged.

To check the threshold on your data before training, render a few clips:

```bash
cd /home/ubuntu/PVOutputPrediction/models/VideoMAEv2
python visualize_sun_blocker_mask.py --lmdb_path /home/ubuntu/PVOutputPrediction/preprocessing/data/uoh_lmdb_minute_224/frames.lmdb --clip_stride_minutes 2 --stem_contains 2018-03-16 --num_clips 4 --out_dir sun_blocker_vis
```

Each PNG shows the augmented frame, the dark pixels in red, the excluded tubelets in magenta, and the encoder input with hidden patches blacked out.
