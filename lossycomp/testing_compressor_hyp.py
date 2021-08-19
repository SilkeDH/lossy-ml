"""Script to test the compressor"""

from lossycomp.dataLoader import DataGenerator, data_preprocessing
from lossycomp.dataLoader import DataGenerator, data_preprocessing
from lossycomp.compress_test import compress
from collections import OrderedDict, defaultdict
import dask
import pickle
import os
import sys
import subprocess
import numpy as np
import zfpy
from lossycomp.constants import data_path

dask.config.set(**{'array.slicing.split_large_chunks': False})

# Load the test data

file = data_path + 'data/ECMWF/1980/*/temperature.nc'
region = "globe"
var = OrderedDict({'t': 1000})

print("Calculating mean z and std...")
z, mean, std = data_preprocessing(file, var, region)

leads = dict(time = 16, longitude=1440, latitude=721, level=1)

samples = 1
batch_size = 1

print("Generating data...")
test_data = DataGenerator(z, samples, leads, batch_size=batch_size, load=True, coords = False, mean= mean, std=std) 

nb_batches = int(samples / batch_size)

model_history = defaultdict(list)

print("Initializing tests...")

for j in range(44):
    j = j +1
    print(j)
    if ((j == 31) or (j==43)):
        pass
    else:
        abs_error = []
        for i in [0.3]:
            compression_factor = []
            for index in range(nb_batches):
                # 1 Channel
                test_batch = test_data.__getitem__(index)
                compressed_data = compress(test_batch[0][0], i, extra_channels = False, verbose = False, method='mask', mode = 'None', convs = 4, hyp = 'hyperparameter/'+j)
                compression_factor.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
                print((test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
            
            abs_error.append(compression_factor)  
                                                     
        model_history['model_' + str(j)].append(abs_error)
    
    pickle.dump({'model': model_history}, open('results/output/CF_hyp.pkl', 'wb'))
