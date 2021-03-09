"""Model functions"""

import scipy.signal
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
import tensorflow.keras as keras
from keras.layers import Input, Conv3D, Conv3DTranspose, Dropout
from keras.layers import LeakyReLU, ReLU, Add
from keras.models import Model
import h5py
import keras.backend as K

def check_gpu():
    """Check which GPUS are available."""
    print(tf.config.list_physical_devices('GPU'))

def mem_limit():
    """Limit TF GPU memory usage usage"""
    tf.config.gpu.set_per_process_memory_growth(True)

def decay_schedule(epoch, lr):
    """Learning rate decay scheduler"""
    if ((epoch != 0) and ((epoch / 100 == 1) and (epoch / 400 == 1)) ):
        lr = lr * 0.1
    return lr

def scheduler(epoch, lr):
    """Learning rate half decay scheduler"""
    if (epoch != 0) and (epoch % 30==0):
        return lr/2
    else:
        return lr

def psnr(y_true, y_pred):
    """Calculates the PSNR
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    psnr = tf.math.log(1 / tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))) / tf.math.log(10.0) * 20
    return psnr

def ssim_loss(y_true, y_pred):
    """Calculates the ssim as a loss function
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def ssim(y_true, y_pred):
    """Calculates the SSIM
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def correlation(x, y): 
    """Calculates the R2
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return  r_num / (r_den + keras.backend.epsilon())
    
def calculate_MSE(y_true, y_pred):
    """Calculates the MSE
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    return np.mean((y_true - y_pred)**2)
    
def calculate_MAE(y_true, y_pred):
    """Calculates the MAE
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    return np.sum(np.absolute(y_true - y_pred))


def mean_squared_error_5(y_true, y_pred):
    """Calculates the MSE if we have more than one channel
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    return K.mean(K.square(y_pred[:,:,:,:,0] - y_true[:,:,:,:,0]), axis=-1)
    
def calculate_MAE_5(y_true, y_pred):
    """Calculates the if we have more than one channel
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    return K.mean(K.abs(y_true[:,:,:,:,0] - y_pred[:,:,:,:,0]))
    
def correlation_5(x, y): 
    """Calculates the correlation constant if we have more than one channel.
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    mx = K.mean(x[:,:,:,:,0])
    my = K.mean(y[:,:,:,:,0])
    xm, ym = x[:,:,:,:,0]-mx, y[:,:,:,:,0]-my
    r_num = K.mean(tf.multiply(xm,ym))        
    r_den = K.std(xm) * K.std(ym)
    return  r_num / (r_den + K.epsilon())
