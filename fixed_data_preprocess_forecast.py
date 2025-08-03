# Fixed version of data preprocessing for forecast
# This script handles resuming from a previous run that was interrupted

import numpy as np
import os
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# folder path and name
project_path = os.getcwd()
data_folder = os.path.join(os.getcwd(),'data')
pred_folder = os.path.join(data_folder,'data_forecast')
pv_data_path = os.path.join(data_folder,'pv_data','pv_output_valid.pkl')

image_name_format = '%Y%m%d%H%M%S'

# Operating parameter
stack_height = 15 # 15 minute
forecast_horizon = 15 # 15 minutes ahead forecast
sampling_interval_all = [2]
output_img_shape = [224, 224, 3]

start_date = datetime.datetime(2017,1,1) #NOTE: Inclusive of start date
end_date = datetime.datetime(2018,1,1) #NOTE: Exclusive of end date (only end up with 2017 data)

# Setting up test set
sunny_day = [(2017,9,15),(2017,10,6),(2017,10,22),(2018,2,16),(2018,6,12),(2018,6,23),(2019,1,25),(2019,6,23),(2019,7,14),(2019,10,14)]
cloudy_day = [(2017,6,24),(2017,9,20),(2017,10,11),(2018,1,25),(2018,3,9),(2018,10,4),(2019,5,27),(2019,6,28),(2019,8,10),(2019,10,19)]

sunny_datetime = [datetime.datetime(day[0],day[1],day[2]) for day in sunny_day]
cloudy_datetime = [datetime.datetime(day[0],day[1],day[2]) for day in cloudy_day]
test_dates = sunny_datetime + cloudy_datetime

def find_idx_with_dates(all_times,test_dates):
    idx=[]
    for test_day in test_dates:
        test_day_end = test_day + datetime.timedelta(days = 1)
        idx+=np.nonzero((all_times>test_day)*(all_times<test_day_end))[0].tolist()
    return idx

def find_time_within_nparray(time_array,time_point):
    probable_idx = np.searchsorted(time_array,time_point)
    
    # If the time point is after all the time in pv_data
    if probable_idx == len(time_array):
        return None   
    
    # See if the time point is actually a match 
    if time_array[probable_idx]== time_point: 
        return probable_idx
        
    else:
        return None

def find_time_within_pdseries(time_array,time_point):
    probable_idx = np.searchsorted(time_array,time_point)
    
    # If the time point is after all the time in pv_data
    if probable_idx == len(time_array):
        return None   
    
    # See if the time point is actually a match 
    if time_array[probable_idx] == time_point: 
        return probable_idx
        
    else:
        return None

# Load in high frequency data
all_times = np.load(os.path.join(data_folder,'data_expanded','all_times_highfreq.npy'), allow_pickle=True)
all_images = np.load(os.path.join(data_folder,'data_expanded','all_images_highfreq.npy'), allow_pickle=True)
pv_data = np.load(pv_data_path, allow_pickle=True)

n_images = all_times.shape[0]
print(f"Total images: {n_images}")

# Create forecast training data file (FIXED VERSION)
sampling_interval = 2
batch_size = 7000
resume_idx = 252000  # Resume from where you left off

# Open file in append mode to resume
with h5py.File('video_prediction_224.h5', 'a') as f:
    
    # Check if datasets already exist (from previous run)
    if 'image_log' in f:
        # Resume mode - datasets already exist
        image_log_ds = f['image_log']
        pv_log_ds = f['pv_log'] 
        pv_pred_ds = f['pv_pred']
        print(f"Resuming from existing datasets. Current size: {image_log_ds.shape[0]}")
    else:
        # First run - create new datasets
        image_log_ds = f.create_dataset(
            'image_log',
            shape=(n_images, stack_height+1, *output_img_shape),
            compression = "gzip",
            dtype='uint8'
        )
        pv_log_ds = f.create_dataset(
            'pv_log',
            shape=(n_images, stack_height+1),
            dtype='float64'
        )
        pv_pred_ds = f.create_dataset(
            'pv_pred',
            shape=(n_images,),
            dtype='float64'
        )
        print("Creating new datasets")

    last_valid_index = 0
    curr_size = image_log_ds.shape[0]  # Start from current size if resuming

    tic = time.process_time()
    for b in range(resume_idx, n_images, batch_size):
        current_batch_size = min(batch_size, n_images-b)
        image_log_batch = np.zeros([current_batch_size,stack_height+1]+output_img_shape,dtype = 'uint8')
        pv_log_batch = np.zeros((current_batch_size,stack_height+1))
        pv_pred_batch = np.zeros(current_batch_size)
        validity_mask = np.ones(current_batch_size,dtype = bool)
        
        sampling_interval_td = datetime.timedelta(minutes = sampling_interval) - datetime.timedelta(seconds=1)
        for i in range(current_batch_size):
            count = b+i
            # See if the specified sampling frequency is met 
            if all_times[count] - all_times[last_valid_index] > sampling_interval_td:

                # Collecting ground truth for predicted value
                pred_time = all_times[count]+datetime.timedelta(minutes=forecast_horizon)
                
                pv_pred_idx = find_time_within_nparray(pv_data.index,pred_time)
                if pv_pred_idx is None:# if prediction ground truth not found
                    validity_mask[i] = False
                else: 
                    pv_pred_batch[i] = pv_data.iloc[pv_pred_idx] 

                # Collecting image log and PV log
                for j in range(stack_height+1):
                    log_time = all_times[count] - datetime.timedelta(minutes = j)
                    # Collecting a stack of image
                    log_time_idx = find_time_within_nparray(all_times,log_time)
                    if log_time_idx is not None:
                        image_log_batch[i,j] = all_images[log_time_idx]
                    else:
                        validity_mask[i] = False
                        break

                    # Collecting a stack of PV value
                    pv_log_idx = find_time_within_nparray(pv_data.index,log_time)
                    # Check if PV value present
                    if pv_log_idx is None:
                        validity_mask[i] = False
                        break
                    else: 
                        pv_log_batch[i,j] = pv_data.iloc[pv_log_idx]    

            else: # if this is in between the sampling points, discard
                validity_mask[i] = False
            
            if validity_mask[i]:
                last_valid_index = i
            
        # Prompt progress of current work
        print('processed {0}/{1} images'.format(b+batch_size,len(all_times)))
            
        # Only pick out the valid time points
        image_log_batch = image_log_batch[validity_mask]
        pv_log_batch = pv_log_batch[validity_mask]
        pv_pred_batch = pv_pred_batch[validity_mask]
        
        # Store information
        image_log_ds[curr_size:curr_size+validity_mask.sum()] = image_log_batch
        pv_log_ds[curr_size:curr_size+validity_mask.sum()] = pv_log_batch
        pv_pred_ds[curr_size:curr_size+validity_mask.sum()] = pv_pred_batch
        curr_size += validity_mask.sum()

        print('For sampling frequency: ',sampling_interval,' minutes')
        f.flush()

    pred_folder_child = os.path.join(pred_folder,'frequency_'+str(sampling_interval))
    print(f"Processing complete. Final dataset size: {curr_size}") 