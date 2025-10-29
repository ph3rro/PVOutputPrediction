# PV Output Forecasting from Sky Images with VideoMAEv2

Data preprocessing scripts for sky-video-PV pairs and modified VideoMAEv2 for regression and PV output forecasting from 15-minute sky videos, with a forecast horizon of 15 minutes.

# Installation

## Prerequisites

- Python 3.13
- NVIDIA GPU with CUDA support (recommended for training)
- CUDA 12.6+ (version compatible with Pytorch)
- Git

## Step 1: Clone the Repository

```bash
git clone https://github.com/ph3rro/PVOutputPrediction
cd PVOutputPrediction
```
## Step 1b: Initialize Git Submodules

The VideoMAEv2 model is included as a git submodule. Initialize and fetch it:

```bash
git submodule init
git submodule update
```

## Step 2: Create Virtual Environment

It's recommended to use a virtual environment to avoid dependency conflicts.

### Windows
```bash
python -m venv pytorch_env
pytorch_env\Scripts\activate
```

### Linux/macOS
```bash
python -m venv pytorch_env
source pytorch_env/bin/activate
```

## Step 3: Install PyTorch with CUDA Support

Install PyTorch first, as it requires specific CUDA versions. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to get the appropriate command for your system.

For CUDA 12.8 (as used in this project):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

For CPU-only (not recommended for training):
```bash
pip install torch torchvision torchaudio
```

## Step 4: Install Core Dependencies

Install the main project dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- Core libraries: numpy, pandas, scipy
- Deep learning: scikit-learn, timm, transformers
- Computer vision: opencv-python, pillow
- Data handling: h5py, hdf5plugin
- Utilities: CRPS, tqdm, tensorboard, matplotlib
- Jupyter notebook support

## Step 5: Install VideoMAE Dependencies (Optional)

If you plan to use the VideoMAE models for video-based prediction:

```bash
cd models/VideoMAEv2
pip install -r requirements-MAE.txt
cd ../..
```

Key VideoMAE dependencies include:
- decord (for video processing)
- einops
- av (PyAV for video I/O)
- timm (PyTorch Image Models)

## Step 6: Verify Installation

Check that PyTorch can access your GPU:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

# VideoMAEv2 Training

## Finetuning

### Windows
```bash
python run_class_finetuning.py --batch_size=3 --lr=1e-3 --num_workers=0 --mixup=0 --cutmix=0
```
### Linux 

## Pretraining

### Windows
```bash
python run_class_pretraining.py
```
### Linux
