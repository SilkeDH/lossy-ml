"""Compressor"""
import bz2
import os
import contextlib, sys
import pickle
import struct
import numpy as np
import xarray as xr
from lossycomp.models import Autoencoder
from lossycomp.dataLoader import chunk_data, merge_data
import lossycomp.arithmetic_coding as arithmetic_coding
from lossycomp.mtf import encode, decode


def compress(array, err_threshold, extra_channels = True, verbose = False):
    """ Compression method
    Args:
    =========
    file: 4D numpy array.
    err_threshold: absolute error
    extra_channels: Consider extra information or not.
    verbose: Show steps and extra information about the compression.
    Returns bytes.
    """
    
    # Check if input is np.float32, if not, cast.
    if array.dtype == np.float64:
        array = np.array(array, dtype = np.float32) # The output of the model is np.float32.

    if extra_channels:
        assert array.shape[3] == 5, "Input should have 5 channels."
        # Loading the model
        if verbose:
            print("Load model...")
        (encoder, decoder, model) = Autoencoder.build(16, 48, 48, 5, filters = (10, 20, 20, 20))
        model.load_weights('../results/models_70_epochs_try/weights/weight.hdf5') # 5 channels
        #Standardizing data
        if verbose:
            print(model.summary())
        if verbose:
            print("Standardizing data...")
        mean = array[:,:,:,0].mean()
        std = array[:,:,:,0].std()
        array_std = array.copy()
        array_std[:,:,:,0] = (array[:,:,:,0] - mean) / (std)
    else:
        assert array.shape[3] == 1, "Input should only have 1 channel."
        # Loading the model
        if verbose:
            print("Load model...")
        (encoder, decoder, model) = Autoencoder.build(16, 48, 48, 1, filters = (10, 20, 20, 20))
        model.load_weights('../results/models_70_epochs_try_1/weights/weight.hdf5') # 1 channel
        if verbose:
            print(model.summary())
        #Standardizing data
        if verbose:
            print("Standardizing data...")
        mean = array.mean()
        std = array.std()
        array_std = (array - mean) / (std)
    
    # Load weights.
    #model.load_weights('../results/model_200_300/weights/params_model_epoch_299.hdf5') #9
    #model.load_weights('../results/model_12/weights/params_model_epoch_199.hdf5') #10,20,20,20
    #model.load_weights('../results/model_8/weights/params_model_epoch_199.hdf5') #10,20
    #model.load_weights('../results/model_9/weights/params_model_epoch_199.hdf5') #10,20,20    
    #model.load_weights('../results/model_channel5//weights/weight.hdf5') # 5 channels
    
    # Load encoder and decoder
    encoder = model.layers[1]
    decoder = model.layers[2]

    # Chunk data.
    if verbose:
        print("Chunking data...")
    chunks = chunk_data(array_std, (16, 48, 48))
      
    # Encoder
    if verbose:
        print("Compressing data...")
    compressed_data = encoder(chunks).numpy()
    
    #Encoder output shape
    comp_data_shape = compressed_data.shape
    
    # Decoder
    decompressed = decoder(compressed_data).numpy()
    
    # Rebuild data
    if verbose:
        print("Merging data...")
    decompressed = merge_data(decompressed, array.shape)
    
    # Unstardardize
    decompressed = (decompressed * std) + (mean)
    
    # Substract error
    if extra_channels:
        error = array[:,:,:,0] - decompressed[:,:,:,0]
    else:
        error = array - decompressed
    
    error = np.expand_dims(error, axis=3)
        
    # TODO: Build a mask instead of setting the values to zero.
    error_neg = np.copy(error)
    error_neg[error_neg >= 0] = 0
    
    error[error < 0] = 0
    
    # Quantization
    if verbose:
        print("Quantizing data...")
    err_threshold = err_threshold * 2.0

    error_neg = abs(error_neg)

    error = np.round(np.divide(error, err_threshold)).astype(np.uint64)
    error_neg = np.round(np.divide(error_neg, err_threshold)).astype(np.uint64)
    
    if verbose:
        print("Encoding data...")
    comp_error = bz2.compress(error, 9)
    
    comp_error_neg = bz2.compress(error_neg, 9)
    
    comp_data = bz2.compress(compressed_data, 9)
    
    # Packing the outputs
    
    out = pickle.dumps([comp_data, comp_data_shape, array.shape, mean, std, comp_error, comp_error_neg, err_threshold])
    
    if verbose:
        print("Compression factor:", array[:,:,:,0].nbytes / len(out))
        print("Latent space compression factor:",  chunks.nbytes / compressed_data.nbytes)
        print("Latent space (%):", len(comp_data) / (len(comp_data)+ len(comp_error)+ len(comp_error_neg)))
        print("Error space (%):", (len(comp_error)+ len(comp_error_neg))/ (len(comp_data)+ len(comp_error)+ len(comp_error_neg)))

    return out 


