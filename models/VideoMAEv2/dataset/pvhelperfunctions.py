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

def cv_split_holdout(split_data, train_ratio=0.9):
    split_data = np.asarray(split_data)
    num_samples = len(split_data)
    indices = np.arange(num_samples)

    # Shuffle ALL indices first with a fixed seed
    np.random.seed(1)
    np.random.shuffle(indices)

    # Then split the shuffled list
    split_idx = int(train_ratio * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    #data_train = split_data[train_indices]
    #data_val = split_data[val_indices]

    return train_indices, val_indices

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

def mc_dropout_ensemble(model, inputs, n_samples=50):
    """
    Runs MC Dropout inference on a PyTorch model using a functional approach.
    """
    # 1. Set model to evaluation mode (fixes Batch Norm layers)
    model.eval()
    
    # 2. Force ONLY Dropout layers to be in 'train' mode (active)
    def enable_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
            
    # Apply this function to every layer in the model
    model.apply(enable_dropout)
    
    # 3. Run the loop
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            # The model will now output different values each time
            predictions.append(model(inputs))
    
    # 4. Stack and Calculate Statistics
    # Stack shape: [n_samples, batch_size, output_dim]
    stack = torch.stack(predictions)
    
    # mu = torch.mean(stack, dim=0)   # The Prediction
    # sigma = torch.std(stack, dim=0) # The Uncertainty
    
    return stack

def calculate_crps(predictions, y_true):
    """
    predictions: shape (ensemble_size, n_samples) -> e.g., (50, 1000)
    y_true: shape (n_samples,) -> e.g., (1000,)
    """
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()

    n_samples = len(y_true)
    crps_list = []

    print(f"Calculating CRPS for {n_samples} points...")
    
    for i in range(n_samples):

        ensemble_distribution = predictions[:, i].flatten()
        
        observation = float(y_true[i])
        
        score, _, _ = pscore(ensemble_distribution, observation).compute()
        
        crps_list.append(float(score))

    return np.mean(crps_list)






# visualization of forecast predictions
dates_test = np.array([dtinfo.date() for dtinfo in times_test])
hours_test = np.array([dtinfo.time() for dtinfo in times_test])

f,axarr = plt.subplots(10,1,sharex=False, sharey = True)
xfmt = mdates.DateFormatter('%H')
fmt_date = datetime.date(2000,1,1)

green = '#8AB8A7'
red = '#8C1515'
blue = '#67AFD2'
grey =  '#B6B1A9'
black = '#2E2D29'
red = '#8C1515'
light_blue = '#67AFD2'
dark_blue = '#016895'
blue = '#4298B5'
black = '#2E2D29'
dark_red = '#820000'
light_red = '#B83A4B'

for i,date in enumerate(cloudy_dates_test):
    ax = axarr[i]
    date_mask = (dates_test == date)
    hours_xaxis= [datetime.datetime.combine(fmt_date, hour) for hour in hours_test[date_mask]] 
    
    rmse = np.sqrt(np.mean(np.square((pv_log_test[date_mask]-prediction_ensemble[date_mask]))))
    mae = np.mean(np.abs((pv_log_test[date_mask]-prediction_ensemble[date_mask])))
    
    ax.plot(hours_xaxis, pv_log_test[date_mask], linewidth = 1,color=black, label = 'Ground truth')
    ax.plot(hours_xaxis, prediction_ensemble[date_mask],linewidth = 1.5,label = 'UNet nowcast mean',color=red,markerfacecolor="None")
    ax.fill_between(hours_xaxis, percent5_prediction[date_mask], percent95_prediction[date_mask], color=light_blue, alpha=0.5, label = '5~95%tile pred.')
    ax.fill_between(hours_xaxis, percent25_prediction[date_mask], percent75_prediction[date_mask], color=blue, alpha=0.75, label = '25~75%tile pred.')
    ax.set_ylabel('PV output (kW)')
    ax.xaxis.set_major_formatter(xfmt)
    ax.text(0.85,0.85,'Cloudy_'+str(i+1), transform=ax.transAxes)
    #ax.text(0.05,0.65,"RMSE: {0:.2f}\nMAE: {1:.2f}\nFS: {2:.2f}%".format(rmse,mae,fs),transform=ax.transAxes)
    ax.text(0.05,0.65,"RMSE: {0:.2f}\nMAE: {1:.2f}".format(rmse,mae),transform=ax.transAxes)

axarr[0].set_ylim(0, 30)
axarr[0].legend(bbox_to_anchor= [.5,1.3], loc = 'upper center', ncol = 4)
axarr[-1].set_xlabel('Hour of day')

f.set_size_inches(10,30)    
plt.show()