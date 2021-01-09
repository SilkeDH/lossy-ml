"""Compressor"""
import pickle
from lossycomp.models import Autoencoder
from lossycomp.utils import load_weights

def compress(filepath, variable):
    """ Compression method
    Args:
    =========
    file: .nc data filepath.
    Returns a Dataset with the compressed values.
    """
    with open('../results/model_11/model-history.pkl', 'rb') as f:   #(32,16,8) = 64
        data = pickle.load(f)
        
    # Load mean and std.
    mean = data['mean']
    std = data['std']
    
    # Load model architecture.
    (encoder, decoder, model) = Autoencoder.build(16, 40, 40, 1, filters = (32, 16, 8))
    
    # Load weights.
    load_weights(model, '../results/model_11/weights/params_model_epoch_199.hdf5')
    
    # Load only encoder
    encoder = model.layers[1]
    
    # Load data
    data = xr.open_mfdataset(filepath, combine='by_coords')
    
    # Transpose data.
    data = data.transpose('time', 'latitude', 'longitude', 'level')
    
    # TODO: Chunk data.
    
    compressed_data = encoder(data).numpy()
    
    return compressed_data