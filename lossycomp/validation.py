import zfpy
import fpzip
import numpy as np
from lossycomp.utils import calculate_R2, calculate_MAE, calculate_MSE

def zfp_compress(data, rate):
    """Compression with ZFP
    Args:
    =================
    data: Numpy array.
    rate: Compression rate.
    Returns compressed data."""
    return zfpy.compress_numpy(data,rate = rate)


def zfp_decompress(bytes):
    """Decompression with ZFP
    Args:
    =================
    data: Compressed data bytes."""
    return zfpy.decompress_numpy(bytes)

def zfp_compress_decompress(data, rate):
    """Does compression and decompression in 
    one step using ZFP.
    Args:
    =================
    data: Numpy array.
    rate: Compression rate.
    Returns all the information about the procedure in a dict."""
    compressed_data = zfp_compress(data, rate)
    decompressed_data = zfp_decompress(compressed_data)
    mse = calculate_MSE(data, decompressed_data)
    mae = calculate_MAE(data, decompressed_data)
    r2 = calculate_R2(data, decompressed_data)
    return {'compressed_data': compressed_data, 
            'input_bytes': data.nbytes, 
            'output_bytes': len(compressed_data), 
            'compression_factor':data.nbytes/len(compressed_data),
            'compression_ratio': len(compressed_data)/data.nbytes,
            'decompressed_data': decompressed_data, 
            'mse': mse,
            'mae': mae,
            'r2': r2}

def fpzip_compress(data, rate):
    """Compression with FPZIP
    Args:
    =================
    data: Numpy array.
    rate: Compression rate.
    Returns compr3essed data."""
    return fpzip.compress(data, rate = rate, order='C') 

def fpzip_decompress(bytes):
    """Decompression with FPZIP
    Args:
    =================
    data: Compressed data bytes.
    Returns decompressed data."""
    return fpzip.decompress(bytes)


def fpzip_compress_decompress(data, rate):
    """Does compression and decompression in 
    one step using FPZIP
    Args:
    =================
    data: Numpy array.
    rate: Compression rate.
    Returns all the information about the procedure in a dic."""
    compressed_data = fpzip_compress(data, rate)
    decompressed_data = fpzip_decompress(compressed_data)
    mse = calculate_MSE(data, decompressed_data)
    mae = calculate_MAE(data, decompressed_data)
    r2 = calculate_R2(data, decompressed_data)
    return {'compressed_data': compressed_data, 
            'input_bytes': data.nbytes, 
            'output_bytes': len(compressed_data), 
            'compression_factor':data.nbytes/len(compressed_data),
            'compression _ratio': len(compressed_data)/data.nbytes,
            'decompressed_data': decompressed_data, 
            'mse': mse,
            'mae': mae,
            'r2': r2}

