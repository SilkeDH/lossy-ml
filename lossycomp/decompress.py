"""Decompressor"""
import pickle
from lossycomp.models import Autoencoder
from lossycomp.dataLoader import merge_data

def decompress(compressed_data, chunks_size):
    """ Decompression method
    Args:
    =========
    compressed_data: Dataset returned by the compression method.
    Returns the reconstructed data as a Dataset.
    """
    with open('../results/model_14/model-history.pkl', 'rb') as f:   
        data = pickle.load(f)
        
    # Load mean and std.
    mean = data['mean']
    std = data['std']
    
    # Load model architecture.
    (encoder, decoder, model) = Autoencoder.build(16, 48, 48, 1, filters = (10, 20, 20, 40))
    
    # Load weights.
    model.load_weights('../results/model_14/weights/params_model_epoch_199.hdf5')
    
    # Load only the dencoder.
    decoder = model.layers[2]
    
    decompressed = decoder(compressed_data).numpy()
    
    # Merge chunks
    decompressed_data = merge_data(decompressed, chunks_size)
    
    decompressed_data = (decompressed_data * std) + mean
    
    return decompressed_data