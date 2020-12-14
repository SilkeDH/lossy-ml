"""Model functions"""

import tensorflow as tf
from tensorflow import Tensor
import tensorflow.keras as keras
from keras.layers import Input, Conv3D, Conv3DTranspose, Dropout
from keras.models import Model

def check_gpu():
    """Check which GPUS are available."""
    return tf.config.list_physical_devices('GPU')

def mem_limit():
    """Limit TF GPU memory usage usage"""
    tf.config.gpu.set_per_process_memory_growth(True)

def decay_schedule(epoch, lr):
    """Learning rate decay scheduler"""
    # decay by 0.1 every 5 epochs; use `% 1` to decay after each epoch
    if (epoch % 10 == 0) and (epoch != 0):
        lr = lr * 0.1
    return lr

def encoder(x: Tensor, filters, kernels, strides, dropout):
    """Builds encoder"""
    for f, k, s in zip(filters[:-1], kernels[:-1], strides[:-1]):
        x = Conv3D(filters = f, kernerls_size = k, activation='relu', strides = s,
                   padding="same", data_format = "channels_last")(x)
        if dropout> 0:
            x = Dropout(dropout)(x)
    return x

def decoder(x: Tensor, filters, kernels, strides, dropout):
    """Builds decoder"""
    for f, k, s in zip(filters[:-1], kernels[:-1], strides[:-1]):
        x = Conv3DTranspose(filters = f, kernerls_size = k, activation='relu', strides = s,
                   padding="same", data_format = "channels_last")(x)
        if dropout> 0:
            x = Dropout(dropout)(x)
            
    x = Conv3DTranspose(filters = f, kernerls_size = k, activation='None', strides = s,
                   padding="same", data_format = "channels_last")(x)  
    return x

def Autoencoder(input_shape, filters, kernels, strides, dropout = 0):
    """Buids Autoencoder
    Args
    =========
    input_shape: Numpy shape from input.
    filters: List of filter sizes. Sizes are int.
    kernels: List of kernel sizes. Sizes are int.
    strides: List of strides sizes. Sizes are int.
    dropout: Float. Indicates dropout percentage.
    Returns compiled TF model.
    """
    inputs = Input(shape=(input_shape))
    
    enc = encoder(inputs, filters, kernels, strides, dropout)
    dec = decoder(enc, filters, kernels, strides, dropout)
    
    model = Model(inputs, dec)
    model.compile(
        optimizer= keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model    
    
