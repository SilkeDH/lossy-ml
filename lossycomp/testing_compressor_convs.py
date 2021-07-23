"""Script to test the compressor"""
import sys
sys.path.insert(0,'/p/home/jusers/donayreholtz1/hdfml/MyProjects/PROJECT_haf/users/donayreholtz1/lossy-ml/')
from lossycomp.dataLoader import DataGenerator, data_preprocessing, split_data
from lossycomp.compress import compress
from collections import OrderedDict, defaultdict
import dask
import pickle
import os
import sys
import subprocess
import numpy as np
import zfpy

dask.config.set(**{'array.slicing.split_large_chunks': False})

# Load the test data
file = '/p/home/jusers/donayreholtz1/hdfml/MyProjects/PROJECT_haf/data/ECMWF/1980/*/temperature.nc'
region = "globe"
var = OrderedDict({'t': 1000})

print("Calculating mean z and std...")
z = data_preprocessing(file, var, region, stats =False)

samples = 10
batch_size = 1
leads = dict(time = 32, longitude=1440, latitude=721, level=1)

train, test = split_data(z, samples, 1, 32, 0.70)

print("Generating data...")
test_data = DataGenerator(z, train, leads, mean = 0, std = 0, batch_size=batch_size, load=True, coords = False, soil = False, standardize = False, shuffle = False) 

nb_batches = int(samples / batch_size)

model_history_3 = defaultdict(list)
model_history_4 = defaultdict(list)
model_history_5 = defaultdict(list)

print("Initializing tests...")
for i in [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]:
    compression_factor_3 = []
    compression_factor_4 = []
    compression_factor_5 = []
    
    latent_space_3 = []
    latent_space_4 = []
    latent_space_5 = []
    
    error_space_3 = []
    error_space_4 = []
    error_space_5 = []
    
    mask_space_3 = []
    mask_space_4 = []
    mask_space_5 = []

    
    for index in range(nb_batches):
        # 1 Channel
        test_batch = test_data.__getitem__(index)
        print('index', i)
        print("Batch ", index, " of ", nb_batches)
        # Model 3 convs
        data_2 = np.expand_dims(test_batch[0][0][:,:,:,0], axis=3)
        compressed_data = compress(data_2, err_threshold=i, extra_channels = False,  method='mask', mode = 'None', convs = 3, hyp = 'model_3_convs' )
        compression_factor_3.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data[0]))
        latent_space_3.append(compressed_data[1])
        error_space_3.append(compressed_data[3])
        mask_space_3.append(compressed_data[2])
        print((test_batch[0][0][:,:,:,0].nbytes/len(compressed_data[0])))

        
        # Model 3 convs
        compressed_data = compress(data_2, err_threshold=i, extra_channels = False,  method='mask', mode = 'None', convs = 4, hyp = 'model_basic_3' )
        compression_factor_4.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data[0]))
        latent_space_4.append(compressed_data[1])
        error_space_4.append(compressed_data[3])
        mask_space_4.append(compressed_data[2])
        print((test_batch[0][0][:,:,:,0].nbytes/len(compressed_data[0])))

        
        # Model 3 convs
        compressed_data = compress(data_2, err_threshold=i, extra_channels = False,  method='mask', mode = 'None', convs = 5, hyp = 'model_5_convs_2' )
        compression_factor_5.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data[0]))
        latent_space_5.append(compressed_data[1])
        error_space_5.append(compressed_data[3])
        mask_space_5.append(compressed_data[2])
        print((test_batch[0][0][:,:,:,0].nbytes/len(compressed_data[0])))

        
        

    model_history_3['cf_' + str(i)].append(compression_factor_3)
    model_history_3['latent_' + str(i)].append(latent_space_3)
    model_history_3['error_' + str(i)].append(error_space_3)
    model_history_3['mask_' + str(i)].append(mask_space_3)
                              
    model_history_4['cf_' + str(i)].append(compression_factor_4)
    model_history_4['latent_' + str(i)].append(latent_space_4)
    model_history_4['error_' + str(i)].append(error_space_4)
    model_history_4['mask_' + str(i)].append(mask_space_4)
    
    model_history_5['cf_' + str(i)].append(compression_factor_5)
    model_history_5['latent_' + str(i)].append(latent_space_5)
    model_history_5['error_' + str(i)].append(error_space_5)
    model_history_5['mask_' + str(i)].append(mask_space_5) 
    
    pickle.dump({'conv_3': model_history_3, 'conv_4':model_history_4, 'conv_5':model_history_5}, open('results/FINAL_2/CF_conv_3_4_5_6.pkl', 'wb'))
