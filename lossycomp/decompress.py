"""Decompressor"""
import pickle
import bz2
import os
import numpy as np
from lossycomp.models import Autoencoder2
from lossycomp.dataLoader import merge_data
import tensorflow as tf
import random
import math
import fpzip

def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    os.environ['TF_CUDNN_DETERMINISTIC']='1'
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)


def decompress(compressed_data, verbose = False):
    """ Decompression method
    Args:
    =========
        compressed_data: Bytes from the encoder.
        verbose: Set to true to print the decompression stages.
    Returns the reconstructed data as a numpy array.
    """
    # Nothing is random
    reset_random_seeds()
    
    #Read parameters
    with open('../results/final_models/model_1/model-history.pkl', 'rb') as f:
        data = pickle.load(f)
    if verbose:
        print(data['parameters'])
        
    filters = data['parameters']['num_filters']
    kernel = data['parameters']['kernel_size']
    lr = data['parameters']['lr']
    res = data['parameters']['res_blocks']
    l2 = data['parameters']['l2']

    if verbose:
        print("Load model...")

    (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 2, convs=4, filters=filters, kernel = kernel, res=res, lr = lr, l2= l2)
    model.load_weights('../results/final_models/model_1/weights/weight.hdf5')

    # Load only the dencoder.
    decoder = model.layers[2]

    # Read the input
    compressed_data, comp_data_shape, array_size, error_shape, mean, std, comp_error, comp_mask, err_threshold = pickle.loads(compressed_data)
    # Decompress the compressed latent space
    if verbose:
        print("Decompressing data...")
    compressed_data =  fpzip.decompress(compressed_data, order='C')
    compressed_data = np.frombuffer(compressed_data, dtype=np.float32).reshape(comp_data_shape)
    compressed_data = np.expand_dims(compressed_data, axis=1)

    # Decode the latent space with the autoencoder
    if verbose:
        print("Decoding data...")

    chunks_set = tf.data.Dataset.from_tensor_slices(compressed_data)
    batch_size = 10
    chunks_set = chunks_set.batch(batch_size)
    decompressed = decoder.predict(chunks_set, steps=math.ceil(compressed_data.shape[0] / batch_size))

    # Merge chunks
    decompressed_data = merge_data(decompressed, array_size)

    # Decompress residuals
    error = bz2.decompress(comp_error)
    error = np.frombuffer(error, dtype=np.int32).reshape(error_shape)
    error_out = np.copy(error)
    error = error.cumsum()

    # Decompress positions
    mask = bz2.decompress(comp_mask)  
    mask = np.frombuffer(mask, dtype=np.byte).reshape((decompressed_data.shape[0]*decompressed_data.shape[1]* decompressed_data.shape[2]*decompressed_data.shape[3], ))
    mask = mask.cumsum()
    mask = mask.reshape(decompressed_data.shape)

    error = error.astype(np.int32) * err_threshold
    error_quan = np.copy(error)
    
    # Replace residuals into positions
    val = np.copy(mask)
    val = np.array(val, dtype = np.float32)
    res = np.where(mask == 2)
    val[val > 0] = error
    val[res] = val[res] * -1.0
    val = val.astype(np.float32)

    # Destandardize the values
    decompressed_data = (decompressed_data * std) + mean

    # Add the error
    if verbose:
        print("Adding the residuals to decompressed data...")
        
    decompressed_data = (decompressed_data + val).astype(np.float32)
    if verbose:
        print("Done.")

    return decompressed_data
