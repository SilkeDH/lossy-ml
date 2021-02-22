"""Decompressor"""
import pickle
import bz2
import numpy as np
from lossycomp.models import Autoencoder
from lossycomp.dataLoader import merge_data

def decompress(compressed_data):
    """ Decompression method
    Args:
    =========
    compressed_data: Bytes from the encoder.
    Returns the reconstructed data as a Dataset.
    """
    
    # Load model architecture.
    (encoder, decoder, model) = Autoencoder.build(16, 48, 48, 1, filters = (10, 20, 20, 20))
    
    # Load weights.
    model.load_weights('../results/model_200_300/weights/params_model_epoch_299.hdf5')
    
    # Load only the dencoder.
    decoder = model.layers[2]
    
    compressed_data, comp_data_shape, array_size, mean, std, comp_error, comp_error_neg, err_threshold = pickle.loads(compressed_data)
    
    compressed_data = bz2.decompress(compressed_data)
    
    compressed_data = np.frombuffer(compressed_data, dtype=np.float32).reshape(comp_data_shape)
    
    decompressed = decoder(compressed_data).numpy()
    
    # Merge chunks
    decompressed_data = merge_data(decompressed, array_size)
    
    error = bz2.decompress(comp_error)
    
    error_neg = bz2.decompress(comp_error_neg)
    
    error = np.frombuffer(error, dtype=np.uint64).reshape(decompressed_data.shape)
    
    error_neg = np.frombuffer(error_neg, dtype=np.uint64).reshape(decompressed_data.shape)

    error = (error * err_threshold) - (error_neg * err_threshold)
    
    decompressed_data = (decompressed_data * std) + mean 

    decompressed_data = decompressed_data + error
    
    return decompressed_data