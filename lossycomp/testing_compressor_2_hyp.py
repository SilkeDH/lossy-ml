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
from lossycomp.constants import Region, REGIONS, data_path
import xarray as xr
from timeit import default_timer as timer

dask.config.set(**{'array.slicing.split_large_chunks': False})

# Load the test data
file = data_path + 'data/ECMWF/1980/*/temperature.nc'
maps = data_path + 'data/ECMWF/1980_single/*/land-sea-mask.nc'
region = "globe"
var = OrderedDict({'t': 1000})

print("Calculating mean z and std...")
z = data_preprocessing(file, var, region, stats =False)

samples = 100
batch_size = 1

region = REGIONS[region]

soil = xr.open_mfdataset(maps, combine='by_coords')
soil = soil.sel(longitude=slice(region.min_lon,region.max_lon),
                 latitude=slice(region.min_lat,region.max_lat))
ds = []
generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
ds.append(soil['lsm'].expand_dims({'level': generic_level}, 1))
soil_d = xr.concat(ds, 'level').transpose('time', 'latitude', 'longitude', 'level')
z["lsm"]=(['time', 'latitude', 'longitude', 'level'], soil_d)

train, test = split_data(z, samples, 1, 32, 0.70)

leads = dict(time = 32, longitude=1440, latitude=721, level=1)

print("Generating data...")
test_data = DataGenerator(z, train, leads, mean = 0, std = 0, batch_size=batch_size, load=True, coords = False, soil = True, standardize = False, shuffle = False) 

nb_batches = int(samples / batch_size)

model_history_model_1 = defaultdict(list)
model_history_model_2 = defaultdict(list)


filename = 'zexact.dat'

for i in [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]:
    compression_factor_model_1 = []
    latent_model_1 = []
    error_model_1 = []
    mask_model_1 = []
    timer_model_1 = []
    time_ae_1 = []
    time_e_1 = []
   
    compression_factor_model_2 = []
    latent_model_2 = []
    error_model_2 = []
    mask_model_2 = []
    timer_model_2 = []
    time_ae_2 = []
    time_e_2 = []
    
    for index in range(nb_batches):
        print('Batch:', index)
        print('Abs. Error:', i)
        test_batch = test_data.__getitem__(index)

        # Model 1
        start = timer()
        compressed_data, latent, error, mask, time_1, time_2 = compress(test_batch[0][0], i, extra_channels = False, verbose = False, method='mask', mode = 'soil', convs = 4, hyp = 'final_models/model_1', enc_lat = 'fpzip')
        end = timer()
        compression_factor_model_1.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('Model 1 CF: ',(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        timer_model_1.append(end - start)
        latent_model_1.append(latent)
        error_model_1.append(error)
        mask_model_1.append(mask)
        time_ae_1.append(time_1)
        time_e_1.append(time_2)
        
         # Model 1
        start = timer()
        compressed_data, latent, error, mask, time_1, time_2 = compress(test_batch[0][0], i, extra_channels = False, verbose = False, method='mask', mode = 'soil', convs = 4, hyp = 'final_models/model_2', enc_lat = 'fpzip')
        end = timer()
        compression_factor_model_2.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('Model 2 CF: ',(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        timer_model_2.append(end - start)
        latent_model_2.append(latent)
        error_model_2.append(error)
        mask_model_2.append(mask)
        time_ae_2.append(time_1)
        time_e_2.append(time_2)

        
    model_history_model_1['cf_' + str(i)].append(compression_factor_model_1)
    model_history_model_1['time_' + str(i)].append(timer_model_1)
    model_history_model_1['latent_' + str(i)].append(latent_model_1)
    model_history_model_1['error_' + str(i)].append(error_model_1)
    model_history_model_1['mask_' + str(i)].append(mask_model_1)
    model_history_model_1['time_ae' + str(i)].append(time_ae_1)
    model_history_model_1['time_e' + str(i)].append(time_e_1)
    
    model_history_model_2['cf_' + str(i)].append(compression_factor_model_2)
    model_history_model_2['time_' + str(i)].append(timer_model_2)
    model_history_model_2['latent_' + str(i)].append(latent_model_2)
    model_history_model_2['error_' + str(i)].append(error_model_2)
    model_history_model_2['mask_' + str(i)].append(mask_model_2)
    model_history_model_2['time_ae' + str(i)].append(time_ae_2)
    model_history_model_2['time_e' + str(i)].append(time_e_2)
    
    pickle.dump({'model_1': model_history_model_1,'model_2': model_history_model_2}, open('results/output/model_1_vs_2_final.pkl', 'wb'))
