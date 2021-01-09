"""Decompressor"""
import pickle
from lossycomp.models import Autoencoder
from lossycomp.utils import load_weights

def compress(compressed_data):
    """ Decompression method
    Args:
    =========
    compressed_data: Dataset returned by the compression method.
    Returns the reconstructed data as a Dataset.
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
    decoder = model.layers[2]
    
    # Chunk data.
    
    decompressed_data = decoder(compressed_data).numpy()
    
    # TODO: Rebuild the data from the chunks.
    
    return decompressed_data