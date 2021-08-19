"""Script to test the compressor"""

from lossycomp.dataLoader import DataGenerator, data_preprocessing, split_data
from lossycomp.compress_test import compress
from collections import OrderedDict, defaultdict
import dask
import pickle
import os
import sys
import subprocess
import numpy as np
import zfpy
from timeit import default_timer as timer
from lossycomp.constants import data_path

dask.config.set(**{'array.slicing.split_large_chunks': False})

# Load the test data
file = data_path + 'data/ECMWF/1980/*/temperature.nc'
region = "globe"
var = OrderedDict({'t': 1000})

print("Calculating mean z and std...")
z = data_preprocessing(file, var, region, stats =False)

samples = 50
batch_size = 1
leads = dict(time = 32, longitude=1440, latitude=721, level=1)

train, test = split_data(z, samples, 1, 32, 0.70)

print("Generating data...")
test_data = DataGenerator(z, train, leads, mean = 0, std = 0, batch_size=batch_size, load=True, coords = False, soil = False, standardize = False, shuffle = False) 

nb_batches = int(samples / batch_size)

model_history_model_1 = defaultdict(list)
model_history_model_2 = defaultdict(list)
model_history_model_3 = defaultdict(list)

filename = 'zexact.dat'

for i in [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]:
    compression_factor_model_1 = []
    compression_factor_model_2 = []
    compression_factor_model_3 = []
    compression_factor_model_4 = []
    
    latent_model_1 = []
    latent_model_2 = []
    latent_model_3 = []
    latent_model_4 = []
    
    error_model_1 = []
    error_model_2 = []
    error_model_3 = []
    error_model_4 = []
    
    mask_model_1 = []
    mask_model_2 = []
    mask_model_3 = []
    mask_model_4 = []
   
    
    compression_factor_zfp = []
    compression_factor_sz = []
    
    timer_model_1 = []
    timer_model_2 = []
    timer_model_3 = []
    timer_model_4 = []
    timer_zfp = []
    timer_sz = []
    
    for index in range(nb_batches):
        
        test_batch = test_data.__getitem__(index)
        #print(test_batch[0][0][:,:,:,0].shape)

        # Model 1
        start = timer()
        compressed_data, latent, error, mask = compress(test_batch[0][0], i, extra_channels = False, verbose = False, method='mask', mode = 'None', convs = 4, hyp = 'hyperparameter/28', enc_lat = 'fpzip')
        end = timer()
        compression_factor_model_1.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('Model 1',(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        timer_model_1.append(end - start)
        latent_model_1.append(latent)
        error_model_1.append(error)
        mask_model_1.append(mask)
        print('Latent: ', latent)
        print('Error: ', error)
        print('Mask: ', mask)

        # Model 2
        start = timer()
        compressed_data, latent, error, mask = compress(test_batch[0][0], i, extra_channels = False, verbose = False, method='mask', mode = 'None', convs = 4, hyp = 'hyperparameter/40', enc_lat = 'fpzip')
        end = timer()
        compression_factor_model_2.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('Model 2',(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        timer_model_2.append(end - start)
        latent_model_2.append(latent)
        error_model_2.append(error)
        mask_model_2.append(mask)

        # Model 3
        start = timer()
        compressed_data, latent, error, mask = compress(test_batch[0][0], i, extra_channels = False, verbose = False, method='mask', mode = 'None', convs = 4, hyp = 'hyperparameter/27', enc_lat = 'fpzip')
        end = timer()
        compression_factor_model_3.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('Model 3',(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        timer_model_3.append(end - start)
        latent_model_3.append(latent)
        error_model_3.append(error)
        mask_model_3.append(mask)
        
        # Model 3
        #start = timer()
        #compressed_data, latent, error, mask = compress(test_batch[0][0], i, extra_channels = False, verbose = False, method='mask', mode = 'None', convs = 4, hyp = 'model_4')
        #end = timer()
        #compression_factor_model_4.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        #print('Model 4',(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        #timer_model_3.append(end - start)
        #latent_model_4.append(latent)
        #error_model_4.append(error)
        #mask_model_4.append(mask)

        
        # SZ
        #test_batch[0][0][:,:,:,0].tofile(filename)
        #start = timer()
        #subprocess.run(['SZ/build/bin/sz', '-z', '-f','-M', 'ABS', '-A', str(i),'-i',  filename, '-3', '32','721' ,'1440'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        #end = timer()
        #compression_factor_sz.append(test_batch[0][0][:,:,:,0].nbytes/os.path.getsize(filename + '.sz'))
        #print('SZ',test_batch[0][0][:,:,:,0].nbytes/os.path.getsize(filename + '.sz'))
        #timer_sz.append(end - start) 

        #ZFP
        #start = timer()
        #compressed_data = zfpy.compress_numpy(test_batch[0][0][:,:,:,0], tolerance = i) 
        #end = timer()
        #compression_factor_zfp.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        #print('ZFP',test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        #timer_zfp.append(end - start) 
        #print("Batch ", index, " of ", nb_batches)

        
    model_history_model_1['cf_' + str(i)].append(compression_factor_model_1)
    model_history_model_1['time_' + str(i)].append(timer_model_1)
    model_history_model_1['latent_' + str(i)].append(latent_model_1)
    model_history_model_1['error_' + str(i)].append(error_model_1)
    model_history_model_1['mask_' + str(i)].append(mask_model_1)
    
    model_history_model_2['cf_' + str(i)].append(compression_factor_model_2)
    model_history_model_2['time_' + str(i)].append(timer_model_2)
    model_history_model_2['latent_' + str(i)].append(latent_model_2)
    model_history_model_2['error_' + str(i)].append(error_model_2)
    model_history_model_2['mask_' + str(i)].append(mask_model_2)
    
    model_history_model_3['cf_' + str(i)].append(compression_factor_model_3)
    model_history_model_3['time_' + str(i)].append(timer_model_3)
    model_history_model_3['latent_' + str(i)].append(latent_model_3)
    model_history_model_3['error_' + str(i)].append(error_model_3)
    model_history_model_3['mask_' + str(i)].append(mask_model_3)
         
        
    #model_history_model_4['cf_' + str(i)].append(compression_factor_model_4)
    #model_history_model_4['time_' + str(i)].append(timer_model_4)
    #model_history_model_4['latent_' + str(i)].append(latent_model_4)
    #model_history_model_4['error_' + str(i)].append(error_model_4)
    #model_history_model_4['mask_' + str(i)].append(mask_model_4)
         
    #model_history_sz['cf_' + str(i)].append(compression_factor_sz)
    #model_history_sz['time_' + str(i)].append(timer_sz)
         
    #model_history_zfp['cf_' + str(i)].append(compression_factor_zfp)
    #model_history_zfp['time_' + str(i)].append(timer_zfp)
    
    #pickle.dump({'model_1': model_history_model_1, 'model_2': model_history_model_2, 'model_3': model_history_model_3, 'model_4': model_history_model_4, 'sz': model_history_sz, 'zfp': model_history_zfp}, open('results/FINAL_2/CF_sz_zfp_model_2.pkl', 'wb'))
    pickle.dump({'model_1': model_history_model_1, 'model_2': model_history_model_2, 'model_3': model_history_model_3}, open('results/output/CF_best_hyp.pkl', 'wb'))
