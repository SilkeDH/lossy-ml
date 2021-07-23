"""Decompressor"""
import pickle
import bz2
import os
import numpy as np
from lossycomp.models import Autoencoder2
from lossycomp.dataLoader import merge_data

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, ReLU, Activation, Input, Reshape, Flatten, Dense, PReLU, ReLU, Add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.optimizers import Adam
import argparse
import dask
import time
import random
import math
from collections import OrderedDict

import fpzip

def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    os.environ['TF_CUDNN_DETERMINISTIC']='1'
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)


def decompress(compressed_data, extra_channels = True, verbose = False,  method='None', mode = 'None', convs = 4, hyp = 1):
    reset_random_seeds()
    """ Decompression method
    Args:
    =========
        compressed_data: Bytes from the encoder.
        extra_channels: Set to false to work only with 1 ch.
        verbose: Set to true to print the decompression stages.
    Returns the reconstructed data as a numpy array.
    """


    with open('../../results/FINAL_2/'+ str(hyp)+'/model-history.pkl', 'rb') as f:
        data = pickle.load(f)
    if verbose:
        print(data['parameters'])
    filters = data['parameters']['num_filters']
    kernel = data['parameters']['kernel_size']
    lr = data['parameters']['lr']
    res = data['parameters']['res_blocks']
    l2 = data['parameters']['l2']

    if extra_channels:
        if verbose:
            print("Load model...")
        if mode == 'None':
            (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 5, convs=4, filters=filters, kernel = kernel, res=res, lr = lr, l2 =l2)
            model.load_weights('../../results/FINAL_2/'+ str(hyp) +'/weights/weight.hdf5')
        elif mode == 'gauss':
            (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 5, convs=4, filters=filters, kernel = kernel, res=res, lr = lr, l2 =l2)
            model.load_weights('results/FINAL_2/'+ str(hyp) +'/weights/weight.hdf5')        
    else:
        if verbose:
            print("Load model...")
        if convs == 3:
            (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 1, convs=3)
            model.load_weights('../../results/FINAL/trial_3_convs/weights/weight.hdf5')
        elif convs == 4:
            if mode == 'None':
                (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 1, convs=4, filters=filters, kernel = kernel, res=res, lr = lr, l2 =l2)
                model.load_weights('../../results/FINAL_2/'+ str(hyp) +'/weights/weight.hdf5')
            elif mode == 'soil':
                (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 2, convs=4, filters=filters, kernel = kernel, res=res, lr = lr, l2 =l2)
                model.load_weights('../../results/FINAL_2/'+ str(hyp) +'/weights/weight.hdf5')
        elif convs == 5:
            (encoder, decoder, model) = Autoencoder2.build(32, 96, 96, 1, convs=5)
            model.load_weights('../../results/FINAL/trial_5_convs/weights/weight.hdf5')

    # Load only the dencoder.
    decoder = model.layers[2]

    # Read the input

    if method == 'None':
        compressed_data, comp_data_shape, array_size, mean, std, comp_error, comp_error_neg, err_threshold = pickle.loads(compressed_data)

        # Decompress the compressed latent space
        if verbose:
            print("Decompressing data...")
        compressed_data = bz2.decompress(compressed_data)

        # Transform to numpy and reshape
        compressed_data = np.frombuffer(compressed_data, dtype=np.float32).reshape(comp_data_shape)

        # Decode the latent space with the autoencoder
        if verbose:
            print("Decoding data...")

        chunks_set = tf.data.Dataset.from_tensor_slices(compressed_data)
        batch_size = 10
        chunks_set = chunks_set.batch(batch_size)
        decompressed = decoder.predict(chunks_set, steps=math.ceil(compressed_data.shape[0] / batch_size))
        #decompressed = decoder(compressed_data).numpy()

        # Merge chunks
        decompressed_data = merge_data(decompressed, array_size)

        # Decompress the error maps
        error = bz2.decompress(comp_error)

        error_neg = bz2.decompress(comp_error_neg)

        error = np.frombuffer(error, dtype=np.uint64).reshape(decompressed_data.shape)

        error_neg = np.frombuffer(error_neg, dtype=np.uint64).reshape(decompressed_data.shape)

        error_p = error * err_threshold

        error_neg = error_neg * err_threshold

        # Dequantize the error values
        if verbose:
            print("Dequantizing values...")

        error = error_p - error_neg

        # Destandardize the values
        decompressed_data2 = (decompressed_data * std) + mean
        decompressed_data = (decompressed_data * std) + mean     

        # Add the error
        if verbose:
            print("Adding the residuals to decompressed data...")
        decompressed_data = decompressed_data + error

        if verbose:
            print("Done.")

        return decompressed_data, decompressed_data2#, error

    elif method == 'mask':
        compressed_data, comp_data_shape, array_size, error_shape, mean, std, comp_error, comp_mask, err_threshold, error_first , mask_first = pickle.loads(compressed_data)
        # Decompress the compressed latent space
        if verbose:
            print("Decompressing data...")
        compressed_data =  bz2.decompress(compressed_data)

        # Transform to numpy and reshape
        compressed_data = np.frombuffer(compressed_data, dtype=np.float32).reshape(comp_data_shape)
        #compressed_data = np.expand_dims(compressed_data, axis=1)

        # Decode the latent space with the autoencoder
        if verbose:
            print("Decoding data...")

        chunks_set = tf.data.Dataset.from_tensor_slices(compressed_data)
        batch_size = 10
        chunks_set = chunks_set.batch(batch_size)
        decompressed = decoder.predict(chunks_set, steps=math.ceil(compressed_data.shape[0] / batch_size))
        #decompressed = decoder(compressed_data).numpy()

        # Merge chunks
        decompressed_data = merge_data(decompressed, array_size)

        # Decompress the error maps
        error = bz2.decompress(comp_error)
        error = np.frombuffer(error, dtype=np.int64).reshape(error_shape)
        error_out = np.copy(error)
        error = np.concatenate(([error_first], error)).cumsum()


        mask = bz2.decompress(comp_mask)
        mask = np.frombuffer(mask, dtype=np.byte).reshape( (decompressed_data.shape[0]*decompressed_data.shape[1]* decompressed_data.shape[2]*decompressed_data.shape[3] -1, ) )
        mask_out = np.copy(mask)
        mask = np.concatenate(([mask_first], mask)).cumsum()
        mask = mask.reshape(decompressed_data.shape)

        error = error.astype(np.int64) * err_threshold

        error_quan = np.copy(error)

        val = np.copy(mask)
        val = np.array(val, dtype = np.float32)
        res = np.where(mask == 2)
        val[val > 0] = error
        val[res] = val[res] * -1.0
        val = val.astype(np.float32)

        # Destandardize the values
        decompressed_data2 = (decompressed_data * std) + mean
        decompressed_data = (decompressed_data * std) + mean

        # Add the error
        if verbose:
            print("Adding the residuals to decompressed data...")
        decompressed_data = decompressed_data + val

        if verbose:
            print("Done.")

        return decompressed_data, decompressed_data2#, error_out, error, val, compressed_data, error_quan, decompressed_data2#, error
