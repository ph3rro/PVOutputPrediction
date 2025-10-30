import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import itertools

import matplotlib.dates as mdates
import numpy.ma as ma
import CRPS.CRPS as pscore
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pickle as pkl

#writer = SummaryWriter()

#%load_ext tensorboard

# day block shuffling of the time stamps, and return shuffled indices
def day_block_shuffle(times_trainval):

    # Only keep the date of each time point
    dates_trainval = np.zeros_like(times_trainval, dtype=datetime.date)
    for i in range(len(times_trainval)):
        dates_trainval[i] = times_trainval[i].date()

    # Chop the indices into blocks, so that each block contains the indices of the same day
    unique_dates = np.unique(dates_trainval)
    blocks = []
    for i in range(len(unique_dates)):
        blocks.append(np.where(dates_trainval == unique_dates[i])[0])

    # shuffle the blocks, and chain it back together
    np.random.seed(1)
    np.random.shuffle(blocks)
    shuffled_indices = np.asarray(list(itertools.chain.from_iterable(blocks)))

    return shuffled_indices

# a cross validation generator function for spliting the dayblock shuffled indices into training and validation
def cv_split_kfold(split_data, fold_index, num_fold):
    '''
    input:
    split_data: the dayblock shuffled indices to be splitted
    fold_index: the ith fold chosen as the validation, used for generating the seed for random shuffling
    num_fold: N-fold cross validation
    output:
    data_train: the train data indices
    data_val: the validation data indices
    '''
    # randomly divides into a training set and a validation set
    num_samples = len(split_data)
    indices = np.arange(num_samples)

    # finding training and validation indices
    val_mask = np.zeros(len(indices), dtype=bool)
    val_mask[int(fold_index / num_fold * num_samples):int((fold_index + 1) / num_fold * num_samples)] = True
    val_indices = indices[val_mask]
    train_indices = indices[np.logical_not(val_mask)]

    # shuffle indices
    np.random.seed(fold_index)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    data_train = split_data[train_indices]
    data_val = split_data[val_indices]

    return data_train,data_val

def cv_split_holdout(split_data, train_ratio = 0.9):
    '''
    input:
    split_data: the dayblock shuffled indices to be splitted
    val_ratio: the ratio of data to be used as validation
    output:
    data_train: the train data indices
    data_val: the validation data indices
    '''
    # randomly divides into a training set and a validation set
    num_samples = len(split_data)
    print("num_samples: ", num_samples)
    indices = np.arange(num_samples)

    # finding training and validation indices
    train_mask = np.zeros(len(indices), dtype=bool)
    train_mask[:int(train_ratio * num_samples)] = True
    train_indices = indices[train_mask]
    val_indices = indices[np.logical_not(train_mask)]

    # shuffle indices
    np.random.seed(1)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    # HDF5 requires indices to be in increasing order for fancy indexing
    data_train = split_data[sorted(train_indices)]
    data_val = split_data[sorted(val_indices)]

    return data_train,data_val

def mask_background(img, center_i = 108, center_j = 110, radius = 108, img_size = 224): # put all background pixels (the ones outside the circle region of sky images) to 0s
	mask = torch.ones((3,img_size,img_size), dtype=bool)
    # These are the hardcoded values for the center and radius of the circle in the sky images
	if img_size == 224:
		center_i = 108
		center_j = 110
		radius = 108
	elif img_size == 64:
		center_i = 30
		center_j = 30
		radius = 31
	for i in range(img_size):
		for j in range(img_size):
			if (i-center_i)**2+(j-center_j)**2>=radius**2:
				mask[:,i,j]=0
	mask_img = img*mask
	return mask_img

def trainval(model, device, loader, optimizer, criterion, mode="train"):
    
    if mode == "train":
        model.train()
    else:
        model.eval()
        
    size = len(loader)
         
    total_loss = 0
    for batch_idx, (image,pv) in enumerate(loader):
        image, pv = image.to(device), pv.to(device)
        
        output = model(image).squeeze()
        loss = criterion(output, pv)
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        
    total_loss = total_loss / size
    return total_loss

