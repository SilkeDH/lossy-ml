"""Compressor"""
import pickle
import xarray as xr
from lossycomp.models import Autoencoder
from lossycomp.dataLoader import chunk_data

def compress(array):
    """ Compression method
    Args:
    =========
    file: 4D numpy array.
    Returns a Dataset with the compressed values.
    """
    with open('../results/model_14/model-history.pkl', 'rb') as f:   #(32,16,8) = 64
        data = pickle.load(f)
        
    # Load mean and std.
    mean = data['mean']
    std = data['std']
    
    # Load model architecture.
    (encoder, decoder, model) = Autoencoder.build(16, 48, 48, 1, filters = (10, 20, 20, 40))
    
    # Load weights.
    model.load_weights('../results/model_14/weights/params_model_epoch_199.hdf5')
    
    # Load only encoder
    encoder = model.layers[1]
    
    # Standardize Data
    
    array = (array - mean) / (std)
    
    print("Data shape:", array.shape)
    
    # Chunk data.
    chunks, num_chunks = chunk_data(array, (16, 48, 48))
    
    print("Chunks shape:", chunks.shape ) # Batch, time, lat, lon, level
    
    compressed_data = encoder(chunks).numpy()
    
    print("Compression factor:", array.nbytes / compressed_data.nbytes)
    
    return compressed_data, num_chunks