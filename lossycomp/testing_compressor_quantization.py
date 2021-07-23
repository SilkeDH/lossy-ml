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
from timeit import default_timer as timer

dask.config.set(**{'array.slicing.split_large_chunks': False})

# Load the test data
file = '/p/home/jusers/donayreholtz1/hdfml/MyProjects/PROJECT_haf/data/ECMWF/1980/*/temperature.nc'
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

model_history_p_n_q = defaultdict(list)
model_history_e_m_q = defaultdict(list)

print("Initializing tests...")
for i in [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]:
    compression_factor_p_n_q = []
    compression_factor_e_m_q = []

    latent_space_p_n_q = []
    latent_space_e_m_q  = []
    
    error_space_p_n_q = []
    error_space_e_m_q = []

    mask_space_p_n_q = []
    mask_space_e_m_q  = []

    timer_model_p_n_q = []
    timer_model_e_m_q = []
    
    for index in range(nb_batches):
        # 1 Channel
        test_batch = test_data.__getitem__(index)
        print('Abs. Error:', i)
        print("Batch: ", index, " of ", nb_batches)
        # Model negative + positive errors
        data_2 = np.expand_dims(test_batch[0][0][:,:,:,0], axis=3)
        start = timer()
        compressed_data = compress(data_2, err_threshold=i, extra_channels = False,  method='None', mode = 'None', convs = 4, hyp = 'model_basic_3' )
        end = timer()
        compression_factor_p_n_q.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data[0]))
        latent_space_p_n_q.append(compressed_data[1])
        error_space_p_n_q.append(compressed_data[3])
        mask_space_p_n_q.append(compressed_data[2])
        timer_model_p_n_q.append(end - start)
        print('CF Negative + positive:', (test_batch[0][0][:,:,:,0].nbytes/len(compressed_data[0])))

        
        # Model error + mask
        start = timer()
        compressed_data = compress(data_2, err_threshold=i, extra_channels = False,  method='mask', mode = 'None', convs = 4, hyp = 'model_basic_3' )
        end = timer()
        compression_factor_e_m_q.append(test_batch[0][0][:,:,:,0].nbytes/len(compressed_data[0]))
        latent_space_e_m_q.append(compressed_data[1])
        error_space_e_m_q.append(compressed_data[3])
        mask_space_e_m_q.append(compressed_data[2])
        timer_model_e_m_q.append(end - start)
        print('CF mask + errores:', (test_batch[0][0][:,:,:,0].nbytes/len(compressed_data[0])))
        

    model_history_p_n_q['cf_' + str(i)].append(compression_factor_p_n_q)
    model_history_p_n_q['latent_' + str(i)].append(latent_space_p_n_q)
    model_history_p_n_q['error1_' + str(i)].append(error_space_p_n_q)
    model_history_p_n_q['error2_' + str(i)].append(mask_space_p_n_q)
    model_history_p_n_q['time_' + str(i)].append(timer_model_p_n_q)
                              
    model_history_e_m_q['cf_' + str(i)].append(compression_factor_e_m_q)
    model_history_e_m_q['latent_' + str(i)].append(latent_space_e_m_q)
    model_history_e_m_q['error1_' + str(i)].append(error_space_e_m_q)
    model_history_e_m_q['error2_' + str(i)].append(mask_space_e_m_q)
    model_history_e_m_q['time_' + str(i)].append(timer_model_e_m_q)
    
    pickle.dump({'conv_pq': model_history_p_n_q, 'conv_me':model_history_e_m_q}, open('results/FINAL_2/CF_quantization.pkl', 'wb'))
