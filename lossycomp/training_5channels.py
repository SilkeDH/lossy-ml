import os
import time
import dask
import pickle
import argparse
import numpy as np
import xarray as xr
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from keras.callbacks import LearningRateScheduler
from lossycomp.dataLoader import DataGenerator, data_preprocessing, split_data, norm_data
from lossycomp.utils import lr_log_reduction, correlation_5, calculate_MAE_5, mean_squared_error_5
from tensorflow.keras.optimizers import Adam
from lossycomp.models import Autoencoder

# ------------------------- Args--------------------------#

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--region", type=str, default="globe",
	help="Region to be trained on.")
ap.add_argument("-e", "--epochs", type=int, default=70,
	help="Number of epochs.")
#ap.add_argument("-v", "--verbose", type=boolean, default=False,
#	help="Verbosing.")
ap.add_argument("-o", "--output", type=str, default="model",
	help="Directory name where results will be saved.")

args = vars(ap.parse_args())
# ------------------------- Start -------------------------#

# create dir
if not(os.path.exists('results/'+args["output"])):
    os.mkdir('results/'+args["output"])
    os.mkdir('results/'+args["output"]+'/weights')

# ------------------------- Models -------------------------#

# Get model info.
#model.summary()

csv_logger = CSVLogger('results/'+ args["output"]+ '/model_history_log.csv', append=True)

checkpoint = ModelCheckpoint('results/'+ args["output"]+ '/weights/weight.hdf5', monitor='loss', verbose=0, save_best_only=False, mode='auto', save_freq=5) 

#early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

learning_rate_start = 1e-3
learning_rate_stop = 1e-4
epo = args["epochs"]
epomin = 30
epostep = 1

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_log_reduction(learning_rate_start, learning_rate_stop, epomin = epomin, epo = epo))

# ------------------------- Data -------------------------#
dask.config.set(**{'array.slicing.split_large_chunks': False})

# Open file with data
file = '/lsdf/kit/scc/projects/abcde/1979/*/ERA5.pl.temperature.nc'
region = args["region"]
var = OrderedDict({'t': 1000})

# Preprocess the data, get mean and variance.
z, mean, std = data_preprocessing(file, var, region)

# Split the data into test and train.
train, test = split_data(z, 0.70)

# Set chunk size.
leads = dict(time = 64, longitude=64, latitude=64, level=1)

# Config
batch_size = 54
verbose = False

# Load data.
dg_train = DataGenerator(train, 36000, leads, batch_size=batch_size, load=True, mean= mean, std=std, standardize = True, coords = True) 
dg_test = DataGenerator(test, 3600, leads, batch_size=batch_size, load=True, mean= mean, std=std, standardize = True, coords = True)

# ------------------------- Training the models -------------------------#
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

#Compile model.
strategy = tf.distribute.MirroredStrategy()
start = time.time()

#with strategy.scope():
    
(encoder, decoder, model) = Autoencoder.build(64, 64, 64, 5, filters = (10, 20, 20, 20))
model.compile(optimizer = Adam(lr=0.001), loss=mean_squared_error_5, metrics=[correlation_5, calculate_MAE_5])
history = model.fit(dg_train, validation_data = dg_test, validation_freq = 5,epochs=args["epochs"], callbacks=[lr_callback, checkpoint, csv_logger])

end = time.time()

pickle.dump({'model': history.history, "time": timer(start, end)}, open('results/'+args["output"]+'/model-history.pkl', 'wb'))

