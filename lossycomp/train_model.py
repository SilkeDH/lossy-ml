import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.optimizers import Adam
import argparse
import dask
import os
import time
import random
import pickle
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import math
from collections import OrderedDict
from dataLoader import DataGenerator, data_preprocessing, split_data
from utils import lr_log_reduction, correlation_5, calculate_MAE_5, mean_squared_error_5, calculate_psnr_5
from lossycomp.models import Autoencoder2
from tensorflow import keras

from lossycomp.constants import Region, REGIONS
import xarray as xr

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-o", "--output", type=str, default="model", help="Directory name where results will be saved.")
ap.add_argument("-c", "--convs", type=int, default ="4", help="Number conv layers.")
ap.add_argument("-f", "--filters", type=int, default="20", help="Number of filters.")
ap.add_argument("-k", "--kernel", type=int, default="3", help="Kernel size.")
ap.add_argument("-lr", "--learning", type=float, default="0.001", help="Learning rate.")
ap.add_argument("-res", "--residual", type=int, default="0", help="Residual blocks.")
ap.add_argument("-l2", "--regularization", type=float, default="0.0001", help="L2 regularization.")
ap.add_argument("-gk", "--gaussian", type=str, default="False", help="Gausian kernel.")
ap.add_argument("-i", "--channels", type=str, default="False", help="Include extra information")
args = vars(ap.parse_args())


if args["gaussian"]== "False":
    hp7= False
else:
    hp7= True

if args["channels"]=="False":
    hp8 = False
else:
    hp8 = True

# create dir
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
hp4 = 0.0001 * math.pow(0.001 / 0.0001, prob)
#hp4 = args["learning"]

# num res blocks
#hp5 = int(prob * (2 - 0) + 0)
hp5 = args["residual"]

# L2
lexp = np.log(0.00005)
rexp = np.log(0.0005)
hp6 = 0.00005 * math.pow(0.0005 / 0.00005, prob)
#hp6 = args["regularization"]

parameters = {'num_convs': hp1, 'num_filters':hp2, 'kernel_size': hp3, 'lr': hp4, 'res_blocks': hp5, 'l2':hp6, 'extra': hp8, 'gaussian': hp7  }
print(parameters)


print("Initializing...", flush=True)

# Activating distributed training
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Loading model
print("Building model...", flush=True)
with strategy.scope():
    if hp8:
        encoder, decoder, model = Autoencoder2.build(16, 48, 48, 5, hp1, hp2, hp3, hp4, hp5, hp6)
    else:
        encoder, decoder, model = Autoencoder2.build(16, 48, 48, 1, hp1, hp2, hp3, hp4, hp5, hp6) 

encoder.summary()
decoder.summary()
model.summary()
# Loading data
dask.config.set(**{'array.slicing.split_large_chunks': False})

print('Loading Data...', flush=True)
file = '/lsdf/kit/scc/projects/abcde/1979/*/ERA5.pl.temperature.nc'
region = "globe"
var = OrderedDict({'t': 1000})

z, mean, std = data_preprocessing(file, var, region)
train, test = split_data(z, 0.70)

leads = dict(time = 16, longitude=48, latitude=48, level=1)

dg_train = DataGenerator(train, 400000, leads, batch_size=400, load=True, mean= mean, std=std, coords = hp8, soil = False, standardize = True) 
dg_test = DataGenerator(test, 40000, leads, batch_size=400, load=True, mean= mean, std=std, coords = hp8, soil = False, standardize = True)

print(dg_train.__getitem__(0)[0][0].shape)

# Compiling model
checkpoint = ModelCheckpoint('results/'+ args["output"]+ '/weights/weight.hdf5', monitor='loss', verbose=0, save_best_only=False, mode='auto', save_freq=5) 
#csv_logger = CSVLogger('results/'+ args["output"] + '/model_history_log.csv', append=True)

learning_rate_start = hp4
learning_rate_stop = hp4* 1e-1
epo = 100
epomin = 10
epostep = 1

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_log_reduction(learning_rate_start, learning_rate_stop, epomin = epomin, epo = epo))

#Compile model.
model.compile(optimizer = Adam(lr=hp4), loss=mean_squared_error_5(hp7), metrics=[mean_squared_error_5(hp7), correlation_5, calculate_psnr_5(mean,std)])

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

start = time.time()
history = model.fit(dg_train, validation_data = dg_test, epochs=100,  validation_freq = 1, callbacks = [checkpoint, lr_callback] )
end = time.time()

pickle.dump({'parameters': parameters ,'model': history.history, "time": timer(start, end)}, open('results/'+args["output"]+'/model-history.pkl', 'wb'))

