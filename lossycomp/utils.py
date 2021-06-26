"""Util functions"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
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
        y_pred: predicted value.
    Returns the PSNR."""
    psnr = tf.math.log(1 / tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))) / tf.math.log(10.0) * 20
    return psnr

def ssim_loss(y_true, y_pred):
    """Calculates the ssim as a loss function
    Args:
    =======
        y_true: real value.
        y_pred: predicted value.
    Returns the SSIM as a loss function."""
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def ssim(y_true, y_pred):
    """Calculates the SSIM
    Args:
    =======
        y_true: real value.
        y_pred: predicted value.
    Returns the SSIM."""
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def correlation(x, y): 
    """Calculates the R2
    Args:
    =======
        y_true: real value.
        y_pred: predicted value.
    Returns the correlation coefficient."""
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
        y_pred: predicted value.
    Returns the MSE."""
    return np.mean((y_true - y_pred)**2)
    
def calculate_MAE(y_true, y_pred):
    """Calculates the MAE
    Args:
    =======
        y_true: real value.
        y_pred: predicted value.
    Returns the MAE."""
    return np.sum(np.absolute(y_true - y_pred))

    
def calculate_MAE_5(y_true, y_pred):
    """Calculates the if we have more than one channel
    Args:
    =======
        y_true: real value.
        y_pred: predicted value.
    Returns the MAE"""
    return K.mean(K.abs(y_true[:,:,:,:,0] - y_pred[:,:,:,:,0]))
    
def correlation_5(x, y): 
    """Calculates the correlation constant if we have more than one channel.
    Args:
    =======
        y_true: real value.
        y_pred: predicted value.
    Returns the correlation coef."""
    mx = K.mean(x[:,:,:,:,0])
    my = K.mean(y[:,:,:,:,0])
    xm, ym = x[:,:,:,:,0]-mx, y[:,:,:,:,0]-my
    r_num = K.mean(tf.multiply(xm,ym))        
    r_den = K.std(xm) * K.std(ym)
    return  r_num / (r_den + K.epsilon())

def lr_log_reduction(learning_rate_start = 1e-3, learning_rate_stop = 1e-5, epo = 10000, epomin= 1000):
    """
    Make learning rate schedule function for linear reduction.
    Args:
    =======
        learning_rate_start (float, optional): Learning rate to start with. The default is 1e-3.
        learning_rate_stop (float, optional): Final learning rate at the end of epo. The default is 1e-5.
        epo (int, optional): Total number of epochs to reduce learning rate towards. The default is 10000.
        epomin (int, optional): Minimum number of epochs at beginning to leave learning rate constant. The default is 1000.
    Returns:
        func: Function to use with LearningRateScheduler.
    """
    def lr_out_log(epoch):
        if(epoch < epomin):
            out = learning_rate_start
        else:
            out = np.exp(float(np.log(learning_rate_start) - (np.log(learning_rate_start)-np.log(learning_rate_stop))/(epo-epomin)*(epoch-epomin)))
        return out
    return lr_out_log


def log10(x):
    numerator = tf.math.log(x)
    return numerator / 10.0


def calculate_psnr_5(mean,std):
    def psnr(y_true, y_pred):
        """Calculates the PSNR
        Args:
        =======
        y_true: real value.
        y_pred: predicted value.
        Returns the PSNR."""
        x = (y_pred[:,:,:,:,0]*std)+ mean
        y = (y_true[:,:,:,:,0]*std)+ mean
        vrange = K.max(y) - K.min(y)
        psnr = 20 *  log10(vrange - (10 * log10(K.mean(K.square(y-x)))))
        return psnr 
    return psnr

def mean_squared_error_5(gauss_kernel):
    """Calculates the MSE if we have more than one channel
    Args:
    =======
        y_true: real value.
        y_pred: predicted value.
    Returns the MSE."""
    def mse(y_true, y_pred):
        if gauss_kernel:
            y_shape = y_pred.shape
            ker = gaussian_kernel(d=y_shape[1] , l=y_shape[2], sig =2)
            ker = ker[None,:,:,:]
            ker = tf.constant(ker, dtype= np.float32)    
            y_1 = y_pred[:,:,:,:,0] *  ker
            y_2 = y_true[:,:,:,:,0] *  ker
            return K.mean(K.square(y_1 - y_2), axis=-1)
        else:
            return K.mean(K.square(y_pred[:,:,:,:,0] - y_true[:,:,:,:,0]), axis=-1)
    return mse


def gaussian_kernel(d=16 , l=48, sig=1.):
    """
    Creates a Gaussian kernel with side length l and a sigma of sig.
    Args:
    ======
        b = batch size
        d = time length
        l = length of the kernel (height and width have same length).
        sig = sigma of the kernel.
    Returns a 4D - Gaussian kernel, the first dimension is just the 2D Gk repeated d times into b batches.
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    kernel = (kernel / np.sum(kernel)) + 1
    
    kernel[kernel>1] += 1
    
    kernel = np.repeat(kernel[np.newaxis, :, :], d, axis=0)
    
    return kernel