"""Script to test the compressor"""

from lossycomp.dataLoader import DataGenerator, data_preprocessing
from lossycomp.compress import compress
from collections import OrderedDict, defaultdict
import dask
import pickle
import os
import sys
import subprocess
import zfpy
from timeit import default_timer as timer


dask.config.set(**{'array.slicing.split_large_chunks': False})

# Load the test data

file = '/lsdf/kit/scc/projects/abcde/1980/*/ERA5.pl.temperature.nc'
region = "globe"
var = OrderedDict({'t': 1000})

z, mean, std = data_preprocessing(file, var, region)

leads = dict(time = 32, longitude=1440, latitude=721, level=1)

samples = 10
batch_size = 1

test_data = DataGenerator(z, samples, leads, batch_size=batch_size, load=True, mean= mean, std=std) 

# Loop trought batches and get the compression factor of each batch.

nb_batches = int(samples / batch_size)

model_history_model = defaultdict(list)
model_history_zfp = defaultdict(list)
model_history_sz = defaultdict(list)


filename = 'zexact.dat'

for i in [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]:
    compression_factor_model = []
    compression_factor_zfp = []
    compression_factor_sz = []
    
    timer_model = []
    timer_zfp = []
    timer_sz = []
    for index in range(nb_batches):
        
        test_batch = test_data.__getitem__(index)
        #print(test_batch[0][0][:,:,:,0].shape)


        start = timer()
        compressed_data = compress(test_batch[0][0], i, extra_channels = False, verbose = False, method='mask', mode = 'None', convs = 4, hyp = 12)
        end = timer()
        compression_factor_model.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('Model',(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        timer_model.append(end - start) 

        # SZ
        test_batch[0][0][:,:,:,0].tofile(filename)
        start = timer()
        subprocess.run(['../SZ/build/bin/sz', '-z', '-f','-M', 'ABS', '-A', str(i),'-i',  filename, '-3', '32','721' ,'1440'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        end = timer()
        compression_factor_sz.append(test_batch[0][0][:,:,:,0].nbytes/os.path.getsize(filename + '.sz'))
        print('SZ',test_batch[0][0][:,:,:,0].nbytes/os.path.getsize(filename + '.sz'))
        timer_sz.append(end - start) 

        #ZFP
        start = timer()
        compressed_data = zfpy.compress_numpy(test_batch[0][0][:,:,:,0], tolerance = i) 
        end = timer()
        compression_factor_zfp.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('ZFP',test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        timer_zfp.append(end - start) 
        print("Batch ", index, " of ", nb_batches)

        
    model_history_model['cf_' + str(i)].append(compression_factor_model)
    model_history_model['time_' + str(i)].append(timer_model)
         
    model_history_sz['cf_' + str(i)].append(compression_factor_sz)
    model_history_sz['time_' + str(i)].append(timer_sz)
         
    model_history_zfp['cf_' + str(i)].append(compression_factor_zfp)
    model_history_zfp['time_' + str(i)].append(timer_zfp)
    
    pickle.dump({'model': model_history_model, 'sz': model_history_sz, 'zfp': model_history_zfp}, open('results/FINAL/CF_sz_zfp_model12_right_2.pkl', 'wb'))

