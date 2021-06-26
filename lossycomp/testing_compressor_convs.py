"""Script to test the compressor"""

from lossycomp.dataLoader import DataGenerator, data_preprocessing
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

file = '/lsdf/kit/scc/projects/abcde/1980/*/ERA5.pl.temperature.nc'
region = "globe"
var = OrderedDict({'t': 1000})

print("Calculating mean z and std...")
z, mean, std = data_preprocessing(file, var, region)

leads = dict(time = 32, longitude=1440, latitude=721, level=1)

samples = 50
batch_size = 1

print("Generating data...")
test_data = DataGenerator(z, samples, leads, batch_size=batch_size, load=True, coords = False, mean= mean, std=std) 

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
    latent_space_3 = []
    
    error_space_4 = []
    error_space_4 = []
    
    latent_space_5 = []
    latent_space_5 = []

    
    for index in range(nb_batches):
        # 1 Channel
        test_batch = test_data.__getitem__(index)
        print('index', index)
        
        # Model 3 convs
        compressed_data = compress(test_batch[0][0], i, extra_channels = False, convs = 3)
        compression_factor_3.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print((test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))

        
        # Model 3 convs
        compressed_data = compress(test_batch[0][0], i, extra_channels = False, convs = 4)
        compression_factor_4.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print((test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))

        
        # Model 3 convs
        compressed_data = compress(test_batch[0][0], i, extra_channels = False, convs = 5)
        compression_factor_5.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print((test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))

        
        print("Batch ", index, " of ", nb_batches)

    model_history_3['cf_' + str(i)].append(compression_factor_3)

                              
    model_history_4['cf_' + str(i)].append(compression_factor_4)

    
    model_history_5['cf_' + str(i)].append(compression_factor_5)

    
    pickle.dump({'conv_3': model_history_3, 'conv_4':model_history_4, 'conv_5':model_history_5}, open('results/FINAL/CF_conv_3_4_5.pkl', 'wb'))
