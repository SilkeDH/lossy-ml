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

samples = 50
batch_size = 1

print("Generating data...")
#test_data = DataGenerator(z, samples, leads, batch_size=batch_size, load=True, mean= mean, std=std) 
test_data_5 = DataGenerator(z, samples, leads, batch_size=batch_size, load=True, coords = True, mean= mean, std=std) 

nb_batches = int(samples / batch_size)

#model_history = defaultdict(list)
model_history_5 = defaultdict(list)
model_history_sz = defaultdict(list)
model_history_zfp = defaultdict(list)

filename = 'zexact.dat'

print("Initializing tests...")
for i in [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5]:
    #compression_factor = []
    compression_factor_5 = []
    compression_factor_sz = []
    compression_factor_zfp = []
    
    #latent_space = []
    latent_space_5 = []
    
    #error_space = []
    error_space_5 = []
    
    for index in range(nb_batches):
        
        # 1 Channel
        
        #test_batch = test_data.__getitem__(index)
        print('index', index)
        
        # Model
        #compressed_data, latent, error = compress(test_batch[0][0], i, extra_channels = False) # [X, batch]
        #compression_factor.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        #print((test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        #latent_space.append(latent)
        #error_space.append(error)
        
        # 5 Channels
        
        test_batch_5 = test_data_5.__getitem__(index)
        
        # Model
        #compressed_data_5  = compress(test_batch_5[0][0], i, extra_channels = True) # [X, batch]
        test_batch_5 = np.array(test_batch_5, dtype = np.float32)
        #compression_factor_5.append(test_batch_5[0][0][:,:,:,0].nbytes/len(compressed_data_5))
        #print('Model',test_batch_5[0][0][:,:,:,0].nbytes/len(compressed_data_5))
        #latent_space_5.append(latent_5)
        #error_space_5.append(error_5)
        
        # SZ
        test_batch_5[0][0][:,:,:,0].tofile(filename)
        subprocess.run(['../SZ/build/bin/sz', '-z', '-f','-M', 'ABS', '-A', str(i),'-i',  filename, '-3', '16','721' ,'1440'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        compression_factor_sz.append(test_batch_5[0][0][:,:,:,0].nbytes/os.path.getsize(filename + '.sz'))
        print('SZ', test_batch_5[0][0][:,:,:,0].nbytes/os.path.getsize(filename + '.sz'))
        
        # ZFP
        compressed_data = zfpy.compress_numpy(test_batch_5[0][0][:,:,:,0], tolerance = i)
        print('ZFP', test_batch_5[0][0][:,:,:,0].nbytes/len(compressed_data))
        compression_factor_zfp.append(test_batch_5[0][0][:,:,:,0].nbytes/len(compressed_data))
        
        print("Batch ", index, " of ", nb_batches)

    #model_history['cf_' + str(i)].append(compression_factor)
    #model_history['latent_' + str(i)].append(latent_space)
    #model_history['error_' + str(i)].append(error_space) 
                              
    #model_history_5['cf_' + str(i)].append(compression_factor_5)
    #model_history_5['latent_' + str(i)].append(latent_space_5)
    #model_history_5['error_' + str(i)].append(error_space_5)
    
    model_history_sz['cf_'+ str(i)].append(compression_factor_sz)
    
    model_history_zfp['cf_'+ str(i)].append(compression_factor_zfp)
    
    pickle.dump({'sz': model_history_sz, 'zfp':model_history_zfp }, open('lossycomp/cf_zfp_vs_sz_history.pkl', 'wb'))
