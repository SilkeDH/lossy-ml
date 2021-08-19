"""Compressor"""
import os
import contextlib, sys
import pickle
import struct
import numpy as np
import xarray as xr
from timeit import default_timer as timer
from lossycomp.models import Autoencoder2
from lossycomp.dataLoader import chunk_data, merge_data
from lossycomp.huffman import getHuffmanCode, decode_huffman

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

import bz2
import fpzip
import zlib
import lzma

from bitstring import *

def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    os.environ['TF_CUDNN_DETERMINISTIC']='1'
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)


def compress(array, err_threshold, extra_channels = True, verbose = False, method='None', mode = 'None', convs = 4, hyp = 1, enc = 'bz2', enc_lat = 'bz2', enc_mask = 'bz2'):
    reset_random_seeds()
    """ Compression algorithm using Deep Convolutional Autoencoders.
    Args:
    =========
        file: 4D numpy array.
        err_threshold: absolute error
        extra_channels: Consider extra information or not.
        verbose: Show steps and extra information about the compression.
        method: 'None' or 'mask', switches quantization method.
        mode: 'None' or 'soil' to indicate if land sea mask is present.
        convs: Number of convolutions.
        hyp: model path stored in /results
        enc: residuals encode method
        enc_lat: latent rep. encode mehod
        enc_mask: positions encode method
    Returns bytes.
    """
    
    # Check if input is np.float32, if not, cast.
    if array.dtype != np.float32:
        array = np.array(array, dtype = np.float32) # The output of the model is np.float32.
        
    #else:
    with open('results/'+ str(hyp)+'/model-history.pkl', 'rb') as f:
        data = pickle.load(f)
    if verbose:
        print(data['parameters'])
        
    filters = data['parameters']['num_filters']
    kernel = data['parameters']['kernel_size']
    lr = data['parameters']['lr']
    res = data['parameters']['res_blocks']
    l2 = data['parameters']['l2']
        
    # Check if error is not zero.
    assert err_threshold!= 0, "The absolute error can't be 0." 

    if extra_channels:
        assert array.shape[3] == 5, "Input should have 5 channels."
        # Loading the model
        if verbose:
            print("Load model...")
        if mode == 'None':
            (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 5, convs=4, filters=filters, kernel = kernel, res=res, lr = lr, l2= l2)
            model.load_weights('results/'+ str(hyp) + '/weights/weight.hdf5')
        elif mode == 'gauss':
            (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 5, convs=4, filters=filters, kernel = kernel, res=res, lr = lr, l2= l2)
            model.load_weights('results/' + str(hyp)  + '/weights/weight.hdf5')        
        #Standardizing data
        if verbose:
            print(model.summary())
            print("Standardizing data...")
        
    else:
        # Loading the model
        if verbose:
            print("Load model...")
        if convs == 3:
            (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 1, convs=3, filters=filters, kernel = kernel, res=res, lr = lr, l2= l2)
            model.load_weights('results/'+ str(hyp) + '/weights/weight.hdf5')
        elif convs == 4:
            if mode == 'None':
                assert array.shape[3] == 1, "Input should only have 1 channel."
                (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 1, convs=4, filters=filters, kernel = kernel, res=res, lr = lr, l2= l2)
                model.load_weights('results/'+ str(hyp) +'/weights/weight.hdf5')
            elif mode == 'soil':
                assert array.shape[3] == 2, "Input should only have 2 channel."
                (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 2, convs=4, filters=filters, kernel = kernel, res=res, lr = lr, l2= l2)
                model.load_weights('../../results/'+ str(hyp) +'/weights/weight.hdf5')
        elif convs == 5:
            (encoder, decoder, model) = Autoencoder2.build(32, 96, 96, 1, convs=5, filters=filters, kernel = kernel, res=res, lr = lr, l2= l2)
            model.load_weights('results/'+ str(hyp) + '/weights/weight.hdf5')   
    if verbose:
        print(model.summary())
        #Standardizing data
        print("Standardizing data...")
  
    mean = np.mean(array[:,:,:,0])
    #std = np.mean(np.std(array[:,:,:,0], axis = 0))
    std =  np.std(array[:,:,:,0])
    array_std = array.copy()
    array_std[:,:,:,0] = (array[:,:,:,0] - mean) / (std)    
    
    # Load encoder and decoder
    encoder = model.layers[1]
    decoder = model.layers[2]

    start = timer()
    # Chunk data.
    if verbose:
        print("Chunking data...")
    if convs == 3:
        chunks = chunk_data(array_std, (16, 48, 48))
    elif convs == 4:
        chunks = chunk_data(array_std, (16, 48, 48))
    elif convs == 5:
        chunks = chunk_data(array_std, (32, 96, 96))
        
    # Encoder
    if verbose:
        print("Compressing data...")
    
    chunks_set = tf.data.Dataset.from_tensor_slices(chunks)
    batch_size = 10
    chunks_set = chunks_set.batch(batch_size)
    compressed_data = encoder.predict(chunks_set, steps=math.ceil(chunks.shape[0] / batch_size))
    
    #compressed_data = encoder(chunks).numpy()
    if verbose:
        print('compressed.')
    
    #Encoder output shape
    comp_data_shape = compressed_data.shape
    
    # Decoder
    if verbose:
        print("Start decompressing")
    chunks_set = tf.data.Dataset.from_tensor_slices(compressed_data)
    batch_size = 10
    chunks_set = chunks_set.batch(batch_size)
    decompressed = decoder.predict(chunks_set, steps=math.ceil(compressed_data.shape[0] / batch_size))
    
    #decompressed = decoder(compressed_data).numpy()
    if verbose:
        print("decompressed")
        print(decompressed.shape)
        print("Merging data...")
        
    decompressed = merge_data(decompressed, array.shape)
    
    # Unstardardize
    decompressed = ((decompressed * std) + (mean)).astype(np.float32)
    
    
    end = timer()
    time_ae = end - start
    # Substract error
    if extra_channels:
        error = array[:,:,:,0] - decompressed[:,:,:,0]
        error = np.expand_dims(error, axis=3)
    elif mode == 'soil':
        error = array[:,:,:,0] - decompressed[:,:,:,0]
        error = np.expand_dims(error, axis=3)
    else:
        error = array.astype(np.float32) - decompressed.astype(np.float32)
        error_q = np.copy(error.astype(np.float32))
        
    if method == 'None':
        error_neg = np.copy(error)
        error_neg[error_neg >= 0] = 0    
        error[error < 0] = 0
        
        # Quantization
        if verbose:
            print("Quantizing data...")
        err_threshold = err_threshold *2.0
        error_neg = abs(error_neg)
        error_shape = error_neg.shape
        error = np.round(np.divide(error, err_threshold)).astype(np.uint64)
        error_neg = np.round(np.divide(error_neg, err_threshold)).astype(np.uint64)
        
        error = error.flatten()
        error_first = error[0]
        error = np.diff(error)
        error = error.astype(np.int64)
        
        error_neg = error_neg.flatten()
        error_first_neg = error_neg[0]
        error_neg = np.diff(error_neg)
        error_neg = error_neg.astype(np.int32)
        
        # Encoding
        if verbose:
            print("Encoding data...")
        comp_error = bz2.compress(error, 9)
        comp_error_neg = bz2.compress(error_neg, 9)
        comp_data = bz2.compress(compressed_data, 9)
        
        latent = (len(comp_data)) / (len(comp_data)+ len(comp_error)+ len(comp_error_neg))
        error_e = (len(comp_error)) / (len(comp_data)+ len(comp_error)+ len(comp_error_neg))
        error_m = (len(comp_error_neg))/ (len(comp_data)+ len(comp_error)+ len(comp_error_neg))
        
        #Output
        out = pickle.dumps([comp_data,comp_data_shape,array.shape,  error_shape, mean, std, comp_error, comp_error_neg, err_threshold, error_first, error_first_neg])
        
        if verbose:
            print("Latent space (%):", len(comp_data) / (len(comp_data)+ len(comp_error)+ len(comp_error_neg)))
            print("Error space (%):", (len(comp_error)+ len(comp_error_neg))/ (len(comp_data)+ len(comp_error)+ len(comp_error_neg)))
            print("Postive Error space (%):", (len(comp_error)) / (len(comp_data)+ len(comp_error)+ len(comp_error_neg)))
            print("Negative Error mask (%):", (len(comp_error_neg))/ (len(comp_data)+ len(comp_error)+ len(comp_error_neg)))
    
            
    elif method == 'mask':
        mask = np.copy(error)
        mask[(mask <= err_threshold) & (mask >= (-1.0 * err_threshold))]= 0 # set within error thres to 0.
        mask[mask>0]= 1
        mask[(mask<0)]= 2
        mask = np.array(mask, dtype = np.byte)

        error = np.abs(error)
        error = error[error > err_threshold]

        err_threshold = (err_threshold) *2.0
        if verbose:
            print("Quantizing data...")

        error_quan = np.copy(error)
        error = np.round(np.divide(error, err_threshold)).astype(np.int32)
        
        quant = np.copy(error)
        print('Quant', quant.shape)
        
        if verbose:
            print("Encoding data...")

        if enc_lat == 'bz2':
            comp_data = bz2.compress(compressed_data, 9)
        elif enc_lat == 'fpzip':
            print('fpzip')
            compressed_data_s = np.squeeze(compressed_data, axis=1)
            comp_data_shape = compressed_data_s.shape
            comp_data = fpzip.compress(compressed_data_s, precision=32, order='C')

        error = error.flatten()
        error_first = error[0]
        error = np.diff(error)
        error = error.astype(np.int32)

        error_shape = error.shape

        start = timer()
        if enc == 'bz2':
            comp_error = bz2.compress(error,9)
        elif enc == 'lzma':
            comp_error = lzma.compress(error)
        else:
            comp_error = zlib.compress(error,9)

        mask = np.array(mask.flatten(), dtype = np.byte)
        mask_val = np.copy(mask)
        print(mask)
        mask_first = mask[0]
        mask = np.diff(mask)
        if enc_mask == 'bz2':
            comp_mask = bz2.compress(mask,9)
        elif enc_mask == 'lzma':
            comp_mask = lzma.compress(mask)
        elif enc_mask == 'zlib':
            comp_mask = zlib.compress(mask,9)
        elif enc_mask == 'huffman':
            mask1 = map(str, mask)
            mask1 = ''.join(list(mask1))
            comp_mask = getHuffmanCode(mask1)
            comp_mask[0] = BitArray(bin=comp_mask[0])
            print(type(comp_mask[0]))

        out = pickle.dumps([comp_data, comp_data_shape, array.shape, error_shape,  mean, std, comp_error, comp_mask, err_threshold,
                            error_first, mask_first])

        latent = (len(comp_data)) / (len(comp_data)+ len(comp_error)+ len(comp_mask))
        error_e = (len(comp_error)) / (len(comp_data)+ len(comp_error)+ len(comp_mask))
        error_m = (len(comp_mask))/ (len(comp_data)+ len(comp_error)+ len(comp_mask))
        end = timer()
        
        time_e = end-start
        
        if verbose:
            print("Latent space (%):", len(comp_data) / (len(comp_data)+ len(comp_error)+ len(comp_mask)))
            print("Error + mask space (%):", (len(comp_error)+ len(comp_mask))/ (len(comp_data)+ len(comp_error)+ len(comp_mask)))
            print("Error space (%):", (len(comp_error)) / (len(comp_data)+ len(comp_error)+ len(comp_mask)))
            print("Error mask (%):", (len(comp_mask))/ (len(comp_data)+ len(comp_error)+ len(comp_mask)))

    if verbose:
            print("Compression factor:", array[:,:,:,0].nbytes / len(out))
            print("Latent space compression factor:",  chunks.nbytes / compressed_data.nbytes)    
            print("Done.")

    return out, latent, error_m, error_e, time_ae, time_e, quant , mask_val# error, mask, decompressed, compressed_data, error_q, error_quan


