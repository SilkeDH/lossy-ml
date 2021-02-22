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


def compress(array, err_threshold):
    """ Compression method
    Args:
    =========
    file: 4D numpy array.
    err_threshold: Absolute error
    Returns a Dataset with the compressed values.
    """
    
    # Load model architecture.
    
    #(encoder, decoder, model) = Autoencoder.build(16, 40, 40, 1, filters = (10, 20))
    (encoder, decoder, model) = Autoencoder.build(16, 40, 40, 1, filters = (10, 20, 20))
    #(encoder, decoder, model) = Autoencoder.build(16, 48, 48, 1, filters = (10,20,20,40))
    #(encoder, decoder, model) = Autoencoder.build(16, 48, 48, 1, filters = (10, 20, 20, 20))
    
    # Load weights.
    #model.load_weights('../results/model_200_300/weights/params_model_epoch_299.hdf5') #9
    #model.load_weights('../results/model_12/weights/params_model_epoch_199.hdf5') #10,20,20,20
    #model.load_weights('../results/model_8/weights/params_model_epoch_199.hdf5') #10,20
    model.load_weights('../results/model_9/weights/params_model_epoch_199.hdf5') #10,20,20
    
    # Load only encoder
    encoder = model.layers[1]
    decoder = model.layers[2]
    
    # Standardize Data
    mean = array.mean()
    std = array.std()
    array_std = (array - mean) / (std)
    
    # Chunk data.
    chunks = chunk_data(array_std, (16, 48, 48))
    
    # Encoder
    compressed_data = encoder(chunks).numpy()
    
    #Encoder output shape
    comp_data_shape = compressed_data.shape
    
    # Decoder
    decompressed = decoder(compressed_data).numpy()
    
    # Rebuild data
    decompressed = merge_data(decompressed, array.shape)
    
    # Unstardardize
    decompressed = (decompressed * std) + (mean)
    
    # Substract error
    error = array - decompressed
    
    # TODO: Build a mask instead of setting the values to zero.
    error_neg = np.copy(error)
    error_neg[error_neg >= 0] = 0
    
    error[error < 0] = 0
    
    # Quantization
    err_threshold = err_threshold * 2.0

    error_neg = abs(error_neg)

    error = np.round(np.divide(error, err_threshold)).astype(np.uint64)
    error_neg = np.round(np.divide(error_neg, err_threshold)).astype(np.uint64)
    
    comp_error = bz2.compress(error, 9)
    
    comp_error_neg = bz2.compress(error_neg, 9)
    
    comp_data = bz2.compress(compressed_data, 9)
    
    # Packing the outputs
    
    out = pickle.dumps([comp_data, comp_data_shape, array.shape, mean, std, comp_error, comp_error_neg, err_threshold])
    
    print("Compression factor:", array.nbytes / len(out))
    
    return out

def write_frequencies(bitout, freqs, size):
    for i in range(size):
        write_int(bitout, 32, freqs.get(i))


def compress_ari(freqs, inp, bitout, size):
    enc = arithmetic_coding.ArithmeticEncoder(32, bitout)
    for i in range(len(inp)):
        symbol = inp[i]
        enc.write(freqs, symbol)
    enc.write(freqs, size)  # EOF
    enc.finish()  # Flush remaining code bits


# Writes an unsigned integer of the given bit width to the given stream.
def write_int(bitout, numbits, value):
    for i in reversed(range(numbits)):
        bitout.write((value >> i) & 1)  # Big endian

        
   # Move to Front
    #error = encode(error, int(error.max() + 1))
    #error_neg = encode(error_neg, int(error_neg.max() + 1))
    
    # Arithmetic coding
    
    #freqs = arithmetic_coding.SimpleFrequencyTable([0] * (int(error.max()+1)))
    #error2 = error.flatten()
    #for i in range(len(error2)):
    #        b = error2[i]
    #        freqs.increment(b)

    #print(freqs)
    #outputfile = "out_error_pos.txt"
    # Read input file again, compress with arithmetic coding, and write output file
    #with contextlib.closing(arithmetic_coding.BitOutputStream(open(outputfile, "wb"))) as bitout:
    #    write_frequencies(bitout, freqs, error.max())
    #    compress_ari(freqs, error2, bitout, error.max())
        
        
    #freqs = arithmetic_coding.SimpleFrequencyTable([0] * (int(error_neg.max()+1)))
    #error2_neg = error_neg.flatten()
    #for i in range(len(error2_neg)):
    #        b = error2_neg[i]
    #        freqs.increment(b)

    #print(freqs)
    #outputfile = "out_error_neg.txt"
    # Read input file again, compress with arithmetic coding, and write output file
    #with contextlib.closing(arithmetic_coding.BitOutputStream(open(outputfile, "wb"))) as bitout:
    #    write_frequencies(bitout, freqs, error_neg.max())
    #    compress_ari(freqs, error2_neg, bitout, error_neg.max())
    
    #error_first_el = error.flatten()[0]
    #error_neg_first_el = error_neg.flatten()[0]
    
    ## np.diff flatten
    #error = np.diff(error.flatten())
    #error_neg = np.diff(error_neg.flatten())
    
    ## np.diff across time.
    #error = np.diff(error, axis = 0)
    #error_neg = np.diff(error_neg, axis = 0)