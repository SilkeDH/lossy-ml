import sys
sys.path.insert(0,'/p/home/jusers/donayreholtz1/hdfml/MyProjects/PROJECT_haf/users/donayreholtz1/lossy-ml/')

import numpy as np
import argparse
import dask
import os
import time
import random
import pickle
import math
import xarray as xr

from collections import OrderedDict, defaultdict
from lossycomp.dataLoader import DataGenerator, data_preprocessing, split_data
from lossycomp.utils import lr_log_reduction, correlation_5, calculate_MAE_5, mean_squared_error_5, psnr_5, timer
from lossycomp.models import Autoencoder2
from lossycomp.constants import Region, REGIONS

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

# Construct the argument parse and parse them
ap = argparse.ArgumentParser()

ap.add_argument("-o", "--output", type=str, default="model", help="Directory name where results will be saved.")
ap.add_argument("-c", "--convs", type=int, default ="4", help="Number conv layers.")
ap.add_argument("-f", "--filters", type=int, default="20", help="Number of filters.")
ap.add_argument("-k", "--kernel", type=int, default="4", help="Kernel size.")
ap.add_argument("-lr", "--learning", type=float, default="0.001", help="Learning rate.")
ap.add_argument("-res", "--residual", type=int, default="0", help="Residual blocks.")
ap.add_argument("-l2", "--regularization", type=float, default="0.0001", help="L2 regularization.")
ap.add_argument("-gk", "--gaussian", type=str, default="False", help="Gausian kernel.")
ap.add_argument("-i", "--channels", type=str, default="False", help="Include extra information")
ap.add_argument("-m", "--mask", type=str, default="False", help="Include land-sea information")

args = vars(ap.parse_args())

# Read if gaussian or extra information modes are true
hp7= False if args["gaussian"]== "False" else True
hp8= False if args["channels"]== "False" else True
hp9= False if args["mask"]== "False" else True

# Create directory to store weigths and paramters
if not(os.path.exists('results/'+args["output"])):
    os.mkdir('results/'+args["output"])
    os.mkdir('results/'+args["output"]+'/weights')

# Generating parameters
#tf.random.set_seed(30)
# Initializing randomness
random_state = random.Random()
prob = float(random_state.random())

# num reducing convs
#hp1 = int(prob * (5 - 4) + 4)
hp1 = args["convs"]

#num filter 
#hp2 = int(prob * (40 - 10) + 10)
hp2 = args["filters"]

# kernel size
#hp3 = int(prob * (8 - 3) + 3)
hp3 = args["kernel"]

# lr
#hp4 = 0.0001 * math.pow(0.001 / 0.0001, prob)
#hp4 = args["learning"]
hp4 = 0.001

# num res blocks
#hp5 = int(prob * (2 - 0) + 0)
hp5 = args["residual"]

# L2
#hp6 = 0.00005 * math.pow(0.0005 / 0.00005, prob)
#hp6 = args["regularization"]
hp6 = 0.00005

# Save model parameters
parameters = {'name': args["output"], 'num_convs': hp1, 'num_filters':hp2, 'kernel_size': hp3, 'lr': hp4, 'res_blocks': hp5, 'l2':hp6, 'extra': hp8, 'gaussian': hp7, 'soil': hp9}

pickle.dump({'parameters': parameters}, open('results/'+args["output"]+'/model-history.pkl', 'wb'))
print(parameters)


print("Initializing...")

# Activating distributed training
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Loading model
print("Building model...", flush=True)
with strategy.scope():
    if hp8: #for extra lat+lon information
        encoder, decoder, model = Autoencoder2.build(16, 48, 48, 5, hp1, hp2, hp3, hp4, hp5, hp6)
    elif hp9: # for soil information
        encoder, decoder, model = Autoencoder2.build(16, 48, 48, 2, hp1, hp2, hp3, hp4, hp5, hp6) 
    else: #no information
        encoder, decoder, model = Autoencoder2.build(16, 48, 48, 1, hp1, hp2, hp3, hp4, hp5, hp6) 

encoder.summary()
decoder.summary()
model.summary()

# Loading data
print('Loading Data...', flush=True)
dask.config.set(**{'array.slicing.split_large_chunks': False})

# Define data files
file = '/p/home/jusers/donayreholtz1/hdfml/MyProjects/PROJECT_haf/data/ECMWF/1979/*/temperature.nc'
maps = '/p/home/jusers/donayreholtz1/hdfml/MyProjects/PROJECT_haf/data/ECMWF/1979_single/*/land-sea-mask.nc'
# Define region of the globe
region = "globe"

# Define pressure level
var = OrderedDict({'t': 1000})

# Get mean, std of the data
z, mean , std = data_preprocessing(file, var, region)

# Pre process data for land-sea mask.
if hp9:
    region = REGIONS[region]
    soil = xr.open_mfdataset(maps, combine='by_coords')
    soil = soil.sel(longitude=slice(region.min_lon,region.max_lon),
                     latitude=slice(region.min_lat,region.max_lat))
    ds = []
    generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
    ds.append(soil['lsm'].expand_dims({'level': generic_level}, 1))
    soil_d = xr.concat(ds, 'level').transpose('time', 'latitude', 'longitude', 'level')
    z["lsm"]=(['time', 'latitude', 'longitude', 'level'], soil_d)

# Split data into training and test and give number of samples.
train, test = split_data(z, 60000, 6000, 16, 0.70)
#train, test = split_data(z, 0.7)

# Define chunk size
leads = dict(time = 16, longitude=48, latitude=48, level=1)

# Load test and train data
dg_train = DataGenerator(z, train, leads, mean, std, batch_size=100, load=True, coords = hp8, soil = hp9, standardize = True, shuffle = True) 
dg_test = DataGenerator(z, test, leads, mean, std, batch_size=100, load=True, coords = hp8, soil = hp9, standardize = True, shuffle = False) 



# Compiling model
checkpoint = ModelCheckpoint('results/'+ args["output"]+ '/weights/weight.hdf5', monitor='loss', verbose=0, save_best_only=False, mode='auto', save_freq=5) 

# Define learning rate callback values
learning_rate_start = hp4
learning_rate_stop = hp4* 1e-1
epo = 100
epomin = 10
epostep = 1
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_log_reduction(learning_rate_start, learning_rate_stop, epomin = epomin, epo = epo))

class SaveValues(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            model_story = defaultdict(list)
            model_story['loss'].append(logs['loss'])
            model_story['mse'].append(logs['mse'])
            model_story['psnr_5'].append(logs['psnr_5'])
            model_story['correlation_5'].append(logs['correlation_5'])
            model_story['lr'].append(logs['lr'])
            pickle.dump({'parameters': parameters, 'model': model_story}, open('results/'+args["output"]+'/model-history.pkl', 'wb'))
        
        else:
            with open('results/'+args["output"]+'/model-history.pkl', 'rb') as fr:
                try:
                    data = pickle.load(fr)
                    data['model']['loss'].append(logs['loss'])
                    data['model']['mse'].append(logs['mse'])
                    data['model']['psnr_5'].append(logs['psnr_5'])
                    data['model']['correlation_5'].append(logs['correlation_5'])
                    data['model']['lr'].append(logs['lr'])

                    if ((epoch+1) % 10) == 0:
                        data['model']['val_loss'].append(logs['val_loss'])
                        data['model']['val_mse'].append(logs['val_mse'])
                        data['model']['val_psnr_5'].append(logs['val_psnr_5'])
                        data['model']['val_correlation_5'].append(logs['val_correlation_5'])
                        
                    pickle.dump({'parameters': parameters , 'model': data['model']}, open('results/'+args["output"]+'/model-history.pkl', 'wb'))

                except EOFError:
                    pass


#Compile model.
model.compile(optimizer = Adam(lr=hp4), loss=mean_squared_error_5(hp7), metrics=[mean_squared_error_5(hp7), correlation_5, psnr_5])

start = time.time()
history = model.fit(dg_train, validation_data = dg_test, epochs=100,  validation_freq = 10, callbacks = [checkpoint, lr_callback,  SaveValues()] )
end = time.time()

with open('results/'+args["output"]+'/model-history.pkl', 'rb') as fr:
    data =  pickle.load(fr)

pickle.dump({'parameters': data['parameters'], 'model': data['model'] ,"time": timer(start, end)}, open('results/'+args["output"]+'/model-history.pkl', 'wb'))

