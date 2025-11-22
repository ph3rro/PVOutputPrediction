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

## Step 2: Create Virtual Environments

Install Python 3.13 and add to PATH.

You will need two separate virtual environments—one for the preprocessing notebooks and one for VideoMAEv2

```bash
python3.13 -m venv preprocessing-env
cd models/VideoMAEv2
python3.13 -m venv VideoMAE-env
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
preprocessing-env/bin/activate
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

# VideoMAEv2 Training

## Finetuning

### CUDA
```bash
python run_class_finetuning.py --batch_size=3 --lr=1e-3 --num_workers=0 --mixup=0 --cutmix=0
```

### CPU (not recommended)
```bash
python run_class_finetuning.py --batch_size=3 --lr=1e-3 --num_workers=0 --mixup=0 --cutmix=0 --device='cpu'
```

## Pretraining

### Windows
```bash
python run_class_pretraining.py
```
### Linux
