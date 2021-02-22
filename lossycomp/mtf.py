"""Move to Front Algortihm"""

from typing import List, Tuple, Union
import numpy as np

def encode(input, dict):
    """
    Args:
    =========
    input: ND-Array of data
    dict: maximum value of the array.
    Returns ND Array processed with MTF.
    """
    dictionary = list(range(dict))
    compressed_data = list()      
    rank = 0
    in_shape = input.shape
    input = input.flatten()
    
    def fnc_c(c):
        rank = dictionary.index(c)    
        ap = rank 
        dictionary.pop(rank)
        dictionary.insert(0, c)
        return ap
    
    compressed_data = np.hstack([fnc_c(c) for c in input])
    return compressed_data.reshape(in_shape)          
    
    
def decode(compressed_data, dict):
    """
    Args:
    =========
    compressed_data: ND Array encoded with MTF.
    dict: Same max value to encode.
    Returns the decoded array.
    """
    dictionary = list(range(dict))
    values = []
    comp_shape = compressed_data.shape
    compressed_data = compressed_data.flatten()
    
    def fnc(rank):
        val_dic = dictionary[rank]
        e = dictionary.pop(rank)
        dictionary.insert(0, e)
        return val_dic
    
    values = np.hstack([fnc(rank) for rank in compressed_data])
    return values