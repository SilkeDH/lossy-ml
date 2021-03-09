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
from lossycomp.utils import check_gpu, decay_schedule, correlation
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
    os.mkdir('results/'+args["output"]+'_1')
    os.mkdir('results/'+args["output"]+'_1'+'/weights')

# ------------------------- Models -------------------------#
(encoder, decoder, model) = Autoencoder.build(16, 48, 48, 5, filters = (10, 20, 20, 20))
(encoder_1, decoder_1, model_1) = Autoencoder.build(16, 48, 48, 1, filters = (10, 20, 20, 20))

# Get model info.
#model.summary()

def mean_squared_error(y_true, y_pred):
     return K.mean(K.square(y_pred[:,:,:,:,0] - y_true[:,:,:,:,0]), axis=-1)
    
def calculate_MAE_5(y_true, y_pred):
    """Calculates de MAE
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    return tf.math.reduce_sum(tf.math.abs(y_true[:,:,:,:,0] - y_pred[:,:,:,:,0]))
    
def correlation_5(x, y):    
    mx = tf.math.reduce_mean(x[:,:,:,:,0])
    my = tf.math.reduce_mean(y[:,:,:,:,0])
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return  r_num / (r_den + K.epsilon())


csv_logger = CSVLogger('results/'+ args["output"]+ '/model_history_log.csv', append=True)
csv_logger_1 = CSVLogger('results/'+ args["output"]+ '_1' + '/model_history_log.csv', append=True)

checkpoint = ModelCheckpoint('results/'+ args["output"]+ '/weights/weight.hdf5', monitor='loss', verbose=0, save_best_only=False, mode='auto', save_freq=1) 
checkpoint_1 = ModelCheckpoint('results/'+ args["output"]+ '_1' +'/weights/weight.hdf5', monitor='loss', verbose=0, save_best_only=False, mode='auto', save_freq=1) 

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

#Compile model.
model.compile(optimizer = Adam(lr=0.001), loss=mean_squared_error, metrics=[correlation_5, calculate_MAE_5])
model_1.compile(optimizer = Adam(lr=0.001), loss=tf.keras.losses.MeanSquaredError(), metrics=[correlation, 'MAE'])


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
leads = dict(time = 16, longitude=48, latitude=48, level=1)

# Config
batch_size = 100
verbose = False

# Load data.
dg_train = DataGenerator(train, 400000, leads, batch_size=batch_size, load=True, mean= mean, std=std, standardize = True, coords = True) 
dg_test = DataGenerator(test, 20000, leads, batch_size=batch_size, load=True, mean= mean, std=std, standardize = True, coords = True)

dg_train_1 = DataGenerator(train, 400000, leads, batch_size=batch_size, load=True, mean= mean, std=std, standardize = True, coords = False) 
dg_test_1 = DataGenerator(test, 20000, leads, batch_size=batch_size, load=True, mean= mean, std=std, standardize = True, coords = False)

# Check if batches have the same data:
assert ((dg_train.__getitem__(0)[0][0][:,:,:,0] ==  dg_train_1.__getitem__(0)[0][0][:,:,:,0]).all()) == True, "Data generators mismatch!"


# ------------------------- Training the models -------------------------#
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

start = time.time()
history_1 = model_1.fit(dg_train_1, validation_data = dg_test_1, epochs=args["epochs"], callbacks=[checkpoint_1, csv_logger_1])

end = time.time()

pickle.dump({'model': history_1.history, "time": timer(start, end)}, open('results/'+args["output"]+'_1'+'/model-history.pkl', 'wb'))

start = time.time()
history = model.fit(dg_train, validation_data = dg_test, epochs=args["epochs"], callbacks=[checkpoint, csv_logger])
end = time.time()

pickle.dump({'model': history.history, "time": timer(start, end)}, open('results/'+args["output"]+'/model-history.pkl', 'wb'))

