import time
import dask
import pickle
import numpy as np
import xarray as xr
import tensorflow as tf
from collections import OrderedDict, defaultdict
from keras.callbacks import LearningRateScheduler
from lossycomp.dataLoader import DataGenerator, data_preprocessing, split_data, norm_data
from lossycomp.utils import check_gpu, Autoencoder, decay_schedule, r2_coef
from lossycomp.plots import mult_plot, single_plot, plot_history

## TODO: Learning rate, saving of plottings, weights, mean and std.

# Load model
model = Autoencoder((16, 48, 48, 1), [10, 20, 20,20], [4, 4, 4, 4], [2, 2, 2, 2])

# Get model info.
model.summary()

#Compile model.
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[r2_coef, 'MAE'])

dask.config.set(**{'array.slicing.split_large_chunks': False})

# Open file with data
file = '/lsdf/kit/scc/projects/abcde/1979/*/ERA5.pl.temperature.nc'
region = "europe"
var = OrderedDict({'t': 1000})

# Preprocess the data, get mean and variance.
z, mean, std = data_preprocessing(file, var, region)

# Split the data into test and train.
train, test = split_data(z, 0.70)

# Set chunk size.
leads = dict(time = 16, longitude=48, latitude=48, level=1)

# Config
batch_size = 100
verbose = True
nb_epochs = 100
model_weights = 'params_model_epoch_'

# Load data.
dg_train = DataGenerator(train, 100000, leads, batch_size=batch_size, load=True, mean= mean, std=std) 
dg_test = DataGenerator(test, 20000, leads, batch_size=batch_size, load=True, mean= mean, std=std)

# Train network
train_history = defaultdict(list)
test_history = defaultdict(list)

for epoch in range(nb_epochs):
    print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
    nb_batches = dg_train.__len__()
    if verbose:
        progress_bar = tf.keras.utils.Progbar(target=nb_batches)
    for index in range(nb_batches):
        if verbose:
            progress_bar.update(index + 1)
        else:
            if index % 100 == 0:
                print('processed {}/{} batches'.format(index + 1, nb_batches))

        train_batch = dg_train.__getitem__(index)

        train_hist = model.train_on_batch(x = train_batch[0], y =  train_batch[1], return_dict = True)
        
    test_hist = model.evaluate(dg_test, return_dict = True, verbose = False)
    
    train_history['train'].append(train_hist)
    train_history['test'].append(test_hist)

    print(' ')

    print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *model.metrics_names))
    print('-' * 65)

    ROW_FMT = '{0:<22s} | {1:<4.4f} | {2:<15.4f} | {3:<5.4f}'
        
    print(ROW_FMT.format('model (train)',
                             *train_hist.values()))
    print(ROW_FMT.format('model (test)',
                             *test_hist.values()))

    # save weights every epoch
    model.save_weights('weights/{0}{1:03d}.hdf5'.format(model_weights, epoch),
                               overwrite=True)

    pickle.dump({'train': train_history}, open('model-history.pkl', 'wb'))
        
  
                    
