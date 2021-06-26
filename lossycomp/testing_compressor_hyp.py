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

leads = dict(time = 16, longitude=1440, latitude=721, level=1)

samples = 10
batch_size = 1

print("Generating data...")
test_data = DataGenerator(z, samples, leads, batch_size=batch_size, load=True, coords = False, mean= mean, std=std) 

nb_batches = int(samples / batch_size)

model_history = defaultdict(list)

print("Initializing tests...")

for j in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
    abs_error = []
    for i in [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]:
        compression_factor = []
        for index in range(nb_batches):
            # 1 Channel
            test_batch = test_data.__getitem__(index)
            compressed_data = compress(test_batch[0][0], i, extra_channels = False, verbose = False, method='None', mode = 'None', convs = 4, hyp = j)
            compression_factor.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
            print((test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
            
        abs_error.append(compression_factor)  
                                                     
    model_history['model_' + str(j)].append(abs_error)
    
    pickle.dump({'model': model_history}, open('results/FINAL/CF_hyp.pkl', 'wb'))
