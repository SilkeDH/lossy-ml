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
    if (epoch % 20 == 0):
        lr = lr * 0.1
    return lr

def encoder(x: Tensor, filters, kernels, strides, dropout):
    """Builds encoder"""
    for f, k, s in zip(filters, kernels, strides):
        x = Conv3D(filters = f, kernel_size = k, activation='relu', strides = s,
                   padding="same", data_format = "channels_last")(x)
        if dropout> 0:
            x = Dropout(dropout)(x)
    return x

def decoder(x: Tensor, filters, kernels, strides, dropout):
    """Builds decoder"""
    kernels = np.flipud(kernels) 
    filters = np.flipud(filters) 
    strides = np.flipud(strides) 
    for f, k, s in zip(filters[1:], kernels[1:], strides[1:]):
        x = Conv3DTranspose(f, (k, k, k), strides = (s, s, s), padding="same", activation='relu')(x)
        if dropout> 0:
            x = Dropout(dropout)(x)
            
    x = Conv3DTranspose(filters = 1, kernel_size = (kernels[-1], kernels[-1] , kernels[-1]),
                        activation=None, strides = (strides[-1], strides[-1], strides[-1]) , padding="same")(x)  
    return x

def Autoencoder(input_shape, filters, kernels, strides, optimizer = keras.optimizers.Adam(), dropout = 0):
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
        optimizer = optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        run_eagerly=True
    )
    return model    

def psnr(y_true, y_pred):
    psnr = tf.math.log(1 / tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))) / tf.math.log(10.0) * 20
    return psnr

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def correlation(x, y):    
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))        
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return  r_num / (r_den + keras.backend.epsilon())

def r2_coef(y_true, y_pred):
    """Calculates de R2
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    #if (y_true.shape != y_pred.shape):
    #    raise ValueError('Dimensions do not match %s and %s' % (y_true.shape, y_pred.shape))
    num = keras.backend.sum((y_true-keras.backend.mean(y_true))*(y_pred-keras.backend.mean(y_pred)))
    div = keras.backend.sqrt(keras.backend.sum((y_true-keras.backend.mean(y_true))*(y_true-keras.backend.mean(y_true))))*keras.backend.sqrt(keras.backend.sum((y_pred-keras.backend.mean(y_pred))*(y_pred-keras.backend.mean(y_pred))))
    return num/(div + keras.backend.epsilon())

#def r2_coef(y_true, y_pred):
#    SS_res =  keras.backend.sum(keras.backend.square(y_true - y_pred))
#    SS_tot = keras.backend.sum(keras.backend.square(y_true-keras.backend.mean(y_true)))
#    return ( 1 - SS_res/(SS_tot + keras.backend.epsilon()))
    
def calculate_MSE(y_true, y_pred):
    """Calculates de MSE
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    return np.mean((y_true - y_pred)**2)
    
def calculate_MAE(y_true, y_pred):
    """Calculates de MAE
    Args:
    =======
    y_true: real value.
    y_pred: predicted value."""
    return np.sum(np.absolute(y_true - y_pred))


"""  This residual Autoencoder hasn't been tested yet.
======================
"""

def residual_block(x: Tensor, filters: int, kernel_size = (3, 3, 3)) -> Tensor:
    y = Conv3D(kernel_size=kernel_size, strides= 1, filters=filters, padding="same")(x)
    y = ReLU()(y)
    y = Conv3D(kernel_size=kernel_size, strides=1, filters=filters, padding="same")(y)
    return y

def res_encoder(x: Tensor):
    kernel_size = (3, 3, 3)
    C = 4
    num_res_blocks = 2
    num_comp_lay = 3
    
    x = Conv3D(kernel_size=kernel_size, strides=1, filters=C, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    first_skip_conn = x
    
    for j in range(num_res_blocks):
        skip_connection = x
        x = residual_block(x, filters=C)
        x = Add()([skip_connection, x])
    
    x = Conv3D(kernel_size=kernel_size, strides= 1, filters=C, padding="same")(x)
    x = Add()([first_skip_conn, x])
    
    C_mult = [2, 2, 1]
    
    for i in range(num_comp_lay):
        C = int(x.get_shape()[-1])
        x = Conv3D(kernel_size=kernel_size, strides= 2, filters=C, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3D(kernel_size=kernel_size, strides= 1, filters=C_mult[i]*C, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)

        
    C = int(x.get_shape()[-1])
    
    x = Conv3D(kernel_size=kernel_size, strides= 1, filters=int(C/2), padding="same")(x)

    return x

def res_decoder(x: Tensor):
    kernel_size = (3, 3, 3)
    num_res_blocks = 2
    nump_comp_lay = 3
    C = int(x.get_shape()[-1])
    
    x = Conv3D(kernel_size=kernel_size, strides=1, filters=2*C,padding="same")(x)
    
    C_div = [1, 2, 2]
    
    for i in range(nump_comp_lay):
        C = int(x.get_shape()[-1])
        C_over_div = int(int(C)/C_div[i])
        x = Conv3D(kernel_size=kernel_size,strides= 1,filters=C_over_div, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3DTranspose(C_over_div, kernel_size=kernel_size, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)

    skip_connection = x
    
    C = 4

    for i in range(num_res_blocks):
        B_skip_connection = x
        x = residual_block(x, filters=C)
        x = Add()([B_skip_connection, x])
        
    x = Conv3D(kernel_size=kernel_size, strides= 1, filters=C, padding="same")(x)
    x = Add()([skip_connection, x])
    
    x = Conv3D(kernel_size=kernel_size, strides= 1, filters= 1, padding="same")(x)

    return x

def ResAutoencoder():
    
    inputs = Input(shape=(24, 40, 40, 1))
    
    enc = res_encoder(inputs)
    dec = res_decoder(enc)
    
    model = Model(inputs, dec)

    model.compile(
        optimizer= keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    return model
