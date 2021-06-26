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
from lossycomp.constants import Region, REGIONS
import xarray as xr

dask.config.set(**{'array.slicing.split_large_chunks': False})

# Load the test data

file = '/lsdf/kit/scc/projects/abcde/1980/*/ERA5.pl.temperature.nc'
maps = '/lsdf/kit/scc/projects/abcde/1980/*/ERA5.sl.land_sea_mask.nc'
region = "globe"
var = OrderedDict({'t': 1000})

print("Calculating mean z and std...")
z, mean, std = data_preprocessing(file, var, region)


region = REGIONS[region]

soil = xr.open_mfdataset(maps, combine='by_coords')
soil = soil.sel(longitude=slice(region.min_lon,region.max_lon),
                 latitude=slice(region.min_lat,region.max_lat))
ds = []
generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
ds.append(soil['lsm'].expand_dims({'level': generic_level}, 1))
soil_d = xr.concat(ds, 'level').transpose('time', 'latitude', 'longitude', 'level')
z["lsm"]=(['time', 'latitude', 'longitude', 'level'], soil_d)



leads = dict(time = 32, longitude=1440, latitude=721, level=1)

samples = 50
batch_size = 1

print("Generating data...")
test_data = DataGenerator(z, samples, leads, batch_size=batch_size, load=True, coords = True, mean= mean, std=std) 


print(test_data.__getitem__(0)[0][0].shape)


nb_batches = int(samples / batch_size)

model_history_basic = defaultdict(list)
model_history_extra = defaultdict(list)
model_history_gauss = defaultdict(list)
model_history_soil = defaultdict(list)

print("Initializing tests...")
for i in [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]:
    compression_factor_basic = []
    compression_factor_extra = []
    compression_factor_gauss = []
    compression_factor_soil = []
    
    latent_space_basic = []
    latent_space_basic = []
    
    error_space_extra = []
    error_space_extra = []
    
    latent_space_gauss = []
    latent_space_gauss= []
    
    latent_space_soil = []
    latent_space_soil= []

    
    for index in range(nb_batches):
        
        
        # 1 Channel
        test_batch = test_data.__getitem__(index)
        print('index', i)
        
        # Model basic convs
        data_2 = np.expand_dims(test_batch[0][0][:,:,:,0], axis=3)
        print('SHAPE BASIC:', data_2.shape)
        #print(data_2.shape)
        compressed_data = compress(data_2, err_threshold=i, extra_channels = False,  method='None', mode = 'None', convs = 4)
        compression_factor_basic.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('CF Basic:', (test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        
        # Model soil convs
        data_2 =  np.concatenate((test_batch[0][0][:,:,:,0:1], test_batch[0][0][:,:,:,5:6]),axis = 3)
        compressed_data = compress(data_2, err_threshold=  i, extra_channels = False,  method='None', mode = 'soil', convs = 4)
        compression_factor_soil.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('CF soil:', (test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))

        print('SHAPE SOIL:', data_2.shape)
        
        
        # Model extra convs
        compressed_data = compress(test_batch[0][0][:,:,:,0:5], err_threshold = i, extra_channels = True,  method='None', mode = 'None', convs = 4)
        compression_factor_extra.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('CF extra:', (test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        print('SHAPE EXTRA:', test_batch[0][0][:,:,:,0:5].shape)
        
        
        # Model gauss convs
        compressed_data = compress(test_batch[0][0][:,:,:,0:5], err_threshold = i, extra_channels = True,  method='None', mode = 'gauss', convs = 4)
        compression_factor_gauss.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data))
        print('CF gauss:', (test_batch[0][0][:,:,:,0].nbytes/len(compressed_data)))
        
        print("Batch ", index, " of ", nb_batches)
        
    model_history_basic['cf_' + str(i)].append(compression_factor_basic)

                              
    model_history_extra['cf_' + str(i)].append(compression_factor_extra)

    
    model_history_gauss['cf_' + str(i)].append(compression_factor_gauss)
    
    model_history_soil['cf_' + str(i)].append(compression_factor_soil)

    
    pickle.dump({'conv_basic': model_history_basic, 'conv_extra':model_history_extra, 'conv_gauss':model_history_gauss, 'conv_soil':model_history_soil}, open('results/FINAL/CF_conv_modes.pkl', 'wb'))
