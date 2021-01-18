import os
import time
import dask
import pickle
import argparse
import numpy as np
import xarray as xr
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from keras.callbacks import LearningRateScheduler
from lossycomp.dataLoader import DataGenerator, data_preprocessing, split_data, norm_data
from lossycomp.utils import check_gpu, decay_schedule, r2_coef
from lossycomp.plots import mult_plot, single_plot, plot_history
from tensorflow.keras.optimizers import Adam
from lossycomp.models import Autoencoder

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--region", type=str, default="globe",
	help="Region to be trained on.")
ap.add_argument("-e", "--epochs", type=int, default=300,
	help="Number of epochs.")
#ap.add_argument("-v", "--verbose", type=boolean, default=False,
#	help="Verbosing.")
ap.add_argument("-o", "--output", type=str, default="model",
	help="Directory name where results will be saved.")
args = vars(ap.parse_args())

# create dir
if not(os.path.exists('results/'+args["output"])):
    os.mkdir('results/'+args["output"])
    os.mkdir('results/'+args["output"]+'/weights')

# Load model
(encoder, decoder, model) = Autoencoder.build(16, 48, 48, 1, filters = (10, 20, 20, 20))

# Get model info.
model.summary()

#Compile model.
model.compile(optimizer = Adam(lr=0.001), loss=tf.keras.losses.MeanSquaredError(), metrics=[r2_coef, 'MAE'])

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
nb_epochs = args["epochs"]
model_weights = 'params_model_epoch_'

# Load data.
dg_train = DataGenerator(train, 100000, leads, batch_size=batch_size, load=True, mean= mean, std=std) 
dg_test = DataGenerator(test, 20000, leads, batch_size=batch_size, load=True, mean= mean, std=std)

# Train network
train_history = defaultdict(list)
test_history = defaultdict(list)

model.load_weights('results/'+args["output"]+'/weights/{0}{1:03d}.hdf5'.format(model_weights, 199))

def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

start = time.time()
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
                print('processed {}/{} batches'.format(index, nb_batches))

        train_batch = dg_train.__getitem__(index)

        train_hist = model.train_on_batch(x = train_batch[0], y =  train_batch[1], return_dict = True)
        
    test_hist = model.evaluate(dg_test, return_dict = True, verbose = False)
    
    train_history['train'].append(train_hist)
    train_history['test'].append(test_hist)
    
    # set learning rate
    if (epoch != 0 and epoch % 20 == 0):
        mean_loss_3 = (train_history['train'][-1]['loss'] + train_history['train'][-2]['loss'] + train_history['train'][-3]['loss']) / 3
        mean_loss_4 = (train_history['train'][-2]['loss'] + train_history['train'][-3]['loss'] + train_history['train'][-4]['loss']) / 3
        if mean_loss_3 > mean_loss_4:
            K.set_value(model.optimizer.learning_rate, K.eval(model.optimizer.lr) / 2 )

    if verbose: 
        print(' ')
        
        print('Learning rate:')
        print(K.eval(model.optimizer.lr))
        
        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *model.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.4f} | {2:<15.4f} | {3:<5.4f}'
        
        print(ROW_FMT.format('model (train)',
                             *train_hist.values()))
        print(ROW_FMT.format('model (test)',
                             *test_hist.values()))

    # save weights every epoch
    model.save_weights('results/'+args["output"]+'/weights/{0}{1:03d}.hdf5'.format(model_weights, epoch),
                               overwrite=True)

end = time.time()
pickle.dump({'model': train_history, "mean": mean, "std": std, "time": timer(start, end)}, open('results/'+args["output"]+'/model-history.pkl', 'wb'))

"""
# Saving of plottings
with open('results/'+args["output"]+'/model-history.pkl', 'rb') as f:
    data = pickle.load(f)

def get_values(data, mode):
    val1 = []
    val2 = []
    val3 = [] 
    for values in data['model'][mode]:
        val1.append(values['loss'])
        val2.append(values['r2_coef'])
        val3.append(values['MAE'])
    return val1, val2, val3


epochs = range(1, 1 + args["epochs"])

loss, r2, mae = get_values(data, 'train')
val_loss, val_r2, val_mae = get_values(data, 'test')

plt.figure(0)
plt.plot(epochs, loss, label = "train")
plt.plot(epochs, val_loss, label = "test")
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training vs Test loss')
plt.legend()
plt.savefig('results/'+args["output"]+'/loss.pdf')

plt.figure(1)
plt.plot(epochs, r2, label = "train")
plt.plot(epochs, val_r2, label = "test")
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.title('Training vs Test R2')
plt.legend()
plt.savefig('results/'+args["output"]+'/r2.pdf')

plt.figure(2)
plt.plot(epochs, mae, label = "train")
plt.plot(epochs, val_mae, label = "test")
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Training vs Test MAE')
plt.legend()
plt.savefig('results/'+args["output"]+'/mae.pdf')
"""