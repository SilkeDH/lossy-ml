"""Script to test the compressor"""

from lossycomp.dataLoader import DataGenerator, data_preprocessing
from lossycomp.compress import compress
from collections import OrderedDict, defaultdict
import dask
import pickle
import os
import sys
import subprocess

dask.config.set(**{'array.slicing.split_large_chunks': False})

# Load the test data

file = '/lsdf/kit/scc/projects/abcde/1980/*/ERA5.pl.temperature.nc'
region = "globe"
var = OrderedDict({'t': 1000})

z, mean, std = data_preprocessing(file, var, region)

leads = dict(time = 16, longitude=1440, latitude=721, level=1)

samples = 200
batch_size = 1
#threshold = 3

test_data = DataGenerator(z, samples, leads, batch_size=batch_size, load=True, mean= mean, std=std) 

# Loop trought batches and get the compression factor of each batch.
#filename = 'zexact.dat'

nb_batches = int(samples / batch_size)

model_history = defaultdict(list)
#sz_history =  defaultdict(list)

for i in [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]:
    #compression_factor_sz = []
    compression_factor = []
    for index in range(nb_batches):
        test_batch = test_data.__getitem__(index)
        
        #Model
        compressed_data = compress(test_batch[0][0], i) # [X, batch]
        compression_factor.append(test_batch[0][0].nbytes/len(compressed_data))
        
        #SZ
        #test_batch[0][0][:,:,:,0].tofile(filename)
        #subprocess.run(['../SZ/build/bin/sz', '-z', '-f','-M', 'ABS', '-A', str(i),'-i',  filename, '-3', '16','721' ,'1440'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        #compression_factor_sz.append(test_batch[0][0][:,:,:,0].nbytes/os.path.getsize(filename + '.sz'))
         
        print("Batch ", index, " of ", nb_batches)

    model_history[str(i)].append(compression_factor)
    #sz_history[str(i)].append(compression_factor_sz)
    
    pickle.dump({'model': model_history}, open('lossycomp/threshold-model-withptimediff-history.pkl', 'wb'))
