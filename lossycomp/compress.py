"""Compressor"""
import os
import pickle
import numpy as np
from lossycomp.models import Autoencoder2
from lossycomp.dataLoader import chunk_data, merge_data
import tensorflow as tf
import random
import math
import bz2
import fpzip

def reset_random_seeds(): 
    os.environ['PYTHONHASHSEED']=str(1)
    os.environ['TF_CUDNN_DETERMINISTIC']='1'
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)


def compress(array, err_threshold, verbose = False):
    """ Compression algorithm using Deep Convolutional Autoencoders.
    Args:
    =========
        array: 4D numpy array.
        err_threshold: absolute error
        extra_channels: Consider extra information or not.
        verbose: Show steps and extra information about the compression.
    Returns bytes.
    """
    reset_random_seeds() #So nothing is random
    
    # Check if error is not zero.
    assert err_threshold!= 0, "The absolute error can't be 0." 
    
    # Check if input is np.float32, if not, cast.
    if array.dtype != np.float32:
        array = np.array(array, dtype = np.float32) # The output of the model is np.float32.
        
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

    assert array.shape[3] == 2, "Input should have 2 channels."
    (encoder, decoder, model) = Autoencoder2.build(16, 48, 48, 2, convs=4, filters=filters, kernel = kernel, res=res, lr = lr, l2= l2)
    model.load_weights('../results/final_models/model_1/weights/weight.hdf5')
        
    if verbose:
        print(model.summary())
        #Standardizing data
        print("Standardizing data...")
  
    mean = np.mean(array[:,:,:,0])  #Mean
    std =  np.std(array[:,:,:,0])   #Std
    array_std = array.copy()
    array_std[:,:,:,0] = (array[:,:,:,0] - mean) / (std)    
    
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
    
    # Getting latent representation
    chunks_set = tf.data.Dataset.from_tensor_slices(chunks) # If the data is too big, we get gpu memory problems
    batch_size = 10
    chunks_set = chunks_set.batch(batch_size)
    compressed_data = encoder.predict(chunks_set, steps=math.ceil(chunks.shape[0] / batch_size))

    if verbose:
        print('Compressed.')
    
    #Encoder output shape
    comp_data_shape = compressed_data.shape
    
    # Decoder
    if verbose:
        print("Start decompressing...")
        
    # Getting reconstructed data    
    chunks_set = tf.data.Dataset.from_tensor_slices(compressed_data)
    batch_size = 10
    chunks_set = chunks_set.batch(batch_size)
    decompressed = decoder.predict(chunks_set, steps=math.ceil(compressed_data.shape[0] / batch_size))
    
    #decompressed = decoder(compressed_data).numpy()
    if verbose:
        print("Decompressed.")
        print(decompressed.shape)
        print("Merging data...")
        
    # Merge chunks
    decompressed = merge_data(decompressed, array.shape)
    
    # Unstardardize
    decompressed = ((decompressed * std) + (mean)).astype(np.float32)

    # Residuals
    error = (array[:,:,:,0] - decompressed[:,:,:,0]).astype(np.float32)

    # Positions
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

    # Quantization
    error_quan = np.copy(error)
    error = np.round(np.divide(error, err_threshold)).astype(np.int32)
    quant = np.copy(error)
     
    if verbose:
        print("Encoding data...")

    # Encode latent representation
    compressed_data_s = np.squeeze(compressed_data, axis=1)
    comp_data_shape = compressed_data_s.shape
    comp_data = fpzip.compress(compressed_data_s, precision=32, order='C')
    
    # Differential coding
    error = error.flatten()
    error_first = error[0]
    error = np.diff(error)
    error = np.concatenate(([error_first], error))
    error = error.astype(np.int32)

    error_shape = error.shape
    
    # Compress residuals
    comp_error = bz2.compress(error,9)

    # Differential coding
    mask = np.array(mask.flatten(), dtype = np.byte)
    mask_val = np.copy(mask)
    mask_first = mask[0]
    mask = np.diff(mask)
    mask = np.concatenate(([mask_first], mask))
    
    # Compress positions
    comp_mask = bz2.compress(mask,9)
        
    out = pickle.dumps([comp_data, comp_data_shape, array.shape, error_shape,  mean, std, comp_error, comp_mask, err_threshold])
   
    if verbose:
        print("Latent space (%):", len(comp_data) / (len(comp_data)+ len(comp_error)+ len(comp_mask)))
        print("Error + mask space (%):", (len(comp_error)+ len(comp_mask))/ (len(comp_data)+ len(comp_error)+ len(comp_mask)))
        print("Error space (%):", (len(comp_error)) / (len(comp_data)+ len(comp_error)+ len(comp_mask)))
        print("Error mask (%):", (len(comp_mask))/ (len(comp_data)+ len(comp_error)+ len(comp_mask)))
        print("Compression factor:", array[:,:,:,0].nbytes / len(out))
        print("Latent space compression factor:",  chunks.nbytes / compressed_data.nbytes)    
        print("Done.")

    return out


