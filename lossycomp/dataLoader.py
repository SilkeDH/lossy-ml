"""Dataset classes and functions."""

import os
import random
import xarray as xr
import numpy as np
from tensorflow import keras
from lossycomp.constants import Region, REGIONS
from lossycomp.encodings import encode_lat, encode_lon

random.seed(1234)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, num_samples, leads, mean, std, batch_size=10, coords = False, load=True):
        """
        Data generator. The samples generated are shuffled.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            data: DataArray
            leads: Determines chunk size on each dimension. 
                   Format dict(time=12, longitude=10, latitude=10, level=1)
            num_samples: Int. Number of random lead chunks.
            batch_size: Int. Batch size.
            load: bool. If True, datadet is loaded into RAM.
            mean: Mean from dataset.
            coords: Include coordinates information.
            std: Standart deviation from dataset.
        """
        self.data = data
        self.num_samples = num_samples
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.leads = leads
        self.coords = coords
        self.check_inputs()
        
        # Normalize
        self.data = (self.data - self.mean) / self.std

        self.check_chunks()
        subset = self.data.isel(
            time = slice(None, self.data.time.size - self.leads["time"]),
            #level = slice(None, self.data.level.size - self.leads["level"]), #TODO: Chunks == input.
            latitude = slice(None, self.data.latitude.size - self.leads["latitude"]),
            longitude = slice(None, self.data.longitude.size - self.leads["longitude"]),
        )
        
        self.subset_shape = subset.time.size, subset.level.size, subset.latitude.size, subset.longitude.size
                
        self.subset_length = int(np.prod(self.subset_shape))
        
        self.on_epoch_end()

        if load: print('Loading data into RAM'); self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.num_samples/self.batch_size)

    def calculateValues(self, ix):
        tix, levix, latix, lonix = np.unravel_index(ix, self.subset_shape)
        subset_selection = dict(
            longitude = slice(lonix, lonix + self.leads["longitude"]),
            latitude = slice(latix, latix + self.leads["latitude"]),
            level = slice(levix, levix + self.leads["level"]),
            time = slice(tix, tix + self.leads["time"]),
        )
        whole = self.data.isel(**subset_selection)
        #whole_data = whole.data
        
        # Add coordinates as extra channels.
        if self.coords:
            lat = whole.coords['latitude'].values
            lon = whole.coords['longitude'].values
            lat_st = np.stack([encode_lat(x) for x in lat])
            lon_st = np.stack([encode_lon(x) for x in lon])
            lat1, lat2 = np.hsplit(lat_st, 2)
            lon1, lon2 = np.hsplit(lon_st, 2)
            xx, yy = np.meshgrid(lon1, lat1)
            xx2, yy2 = np.meshgrid(lon2, lat2)
            coords_lat = np.concatenate([[xx]] * len(whole.time), axis=0)
            coords_lon = np.concatenate([[yy]] * len(whole.time), axis=0)
            coords_lat1 = np.concatenate([[xx2]] * len(whole.time), axis=0)
            coords_lon1 = np.concatenate([[yy2]] * len(whole.time), axis=0)
            coords_lat = np.expand_dims(coords_lat, axis=3)
            coords_lon = np.expand_dims(coords_lon, axis=3)
            coords_lat1 = np.expand_dims(coords_lat1, axis=3)
            coords_lon1 = np.expand_dims(coords_lon1, axis=3)
            whole_data =  np.concatenate((whole_data, coords_lat, coords_lon, coords_lat1, coords_lon1 ),axis = 3)
        return whole
    
    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size : (i + 1) * self.batch_size]
        x = np.stack([self.calculateValues(i).data for i in idxs], axis = 0)
        return x, x
    
    def info_extra(self, ix):
        tix, levix, latix, lonix = np.unravel_index(ix, self.subset_shape)
        subset_selection = dict(
            longitude = slice(lonix, lonix + self.leads["longitude"]),
            latitude = slice(latix, latix + self.leads["latitude"]),
            level = slice(levix, levix + self.leads["level"]),
            time = slice(tix, tix + self.leads["time"]),
        )
        return subset_selection
    
    def info(self, i):
        idxs = self.idxs[i * self.batch_size : (i + 1) * self.batch_size]
        info = np.stack([self.info_extra(i) for i in idxs], axis = 0)
        return info 
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = random.sample(range(0, self.subset_length), self.num_samples)
            
    def check_inputs(self):
        assert isinstance(self.data, xr.core.dataarray.DataArray)
        assert isinstance(self.leads, dict)
        assert all(var in self.leads for var in ['longitude', 'time', 'latitude', 'level'])
        
    def check_chunks(self):
        tim_diff = self.leads["time"] - len(self.data.time)
        lat_diff = self.leads["latitude"] - len(self.data.latitude)
        lon_diff = self.leads["longitude"] - len(self.data.longitude)
        lev_diff = self.leads["level"] - len(self.data.level)
        if (tim_diff >= 0 or lat_diff>= 0 or lon_diff >= 0): #or lev_diff > 0)
            raise ValueError("Chunk size can't be equal or greater than actual length.")
            
    
def data_preprocessing(path, var, region):
    """ Returns preprocessed data, mean and std.
    Arguments
    =========
    path: Path to the netcdf file.
    var: Dictionary of the form {'var': level}. Use None for level if data is of single level.
    region: Region to be cropped. Boundary region over lat,lon. 
    Predefined regions set in `constants` module.
    """

    assert isinstance(region, (str, Region))
    if isinstance(region, str) and region not in REGIONS.keys():
        raise KeyError("Region unknown!")
    else:
        region = REGIONS[region] 
        
    assert isinstance(var, dict)
    ds = xr.open_mfdataset(path, combine='by_coords')
    assert hasattr(ds, list(var.keys())[0])
    ds.close()
    
    z = xr.open_mfdataset(path, combine='by_coords')
    data = z.sel(longitude=slice(region.min_lon,region.max_lon),
                 latitude=slice(region.min_lat,region.max_lat))
    ds = []
    generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
    for v, levels in var.items():
        try:
            ds.append(data[v].sel(level=levels))
        except ValueError:
            ds.append(data[v].expand_dims({'level': generic_level}, 1))

    data = xr.concat(ds, 'level').transpose('time', 'latitude', 'longitude', 'level')
    #mean = data.mean(('time', 'latitude', 'longitude')).compute() 
    #std = data.std('time').mean(('latitude', 'longitude')).compute() 
    return (data)#, mean.data[0], std.data[0])
    
    
def split_data(data, percentage):
    """ Returns splitted train-test data depending
    on specified dimension.
    Arguments
    =========
    data: Dataarray.
    percentage: % of data in train data in decimal (70% = 0.7).
    """
    idx = np.arange(len(data.time))
    np.random.shuffle(idx)
    train_idx =idx[0: int(len(idx)*percentage)]
    test_idx = idx[len(train_idx):len(idx)]

    data_train = data.isel(time=np.sort(train_idx))
    data_test = data.isel(time=np.sort(test_idx))
    return (data_train, data_test)
    
def norm_data(data, mean, std):
    norm_data = data * std + mean
    return (norm_data)
   
def chunk_data(data, size):
    """
    Chunks data into (N, t, lat, lon, level)
    Args:
    data: 4D - Numpy array.
    size: (t, lat, lon, level)
    
    Returns N chunks with size (t, lat, long, level)
    """
    assert (data.shape[0] >= size[0] and data.shape[1] >= size[1] and data.shape[2] >= size[2]), 'Chunks size cannot be larger than the data size.'
    
    chunk_num = (int(data.shape[0]/size[0] ), int(data.shape[1]/size[1]  ), int(data.shape[2] /size[2]))
    res_chunk = (data.shape[0]%size[0], data.shape[1]%size[1] ,data.shape[2] %size[2])
    
    length = int(np.prod(chunk_num))    
    
    x = np.stack([calculateChunks(data, i, size, chunk_num) for i in range(length)], axis = 0)

    return x, chunk_num

def calculateChunks(data, i, size, chunk_num):
    tix, latix, lonix = np.unravel_index(i, chunk_num )
    return data[tix*size[0]: tix*size[0] + size[0], latix*size[1]: latix*size[1] + size[1], lonix*size[2]: lonix*size[2] + size[2]]


def merge_data(data, num_chunks):
    """
    Merges data into (t, lat, lon, level)
    Args:
    data: 5D - Numpy array.
    size: (t, lat, lon, level)
    
    Returns complete data (t, lat, long, level)
    """
    x = np.concatenate([  np.concatenate([np.concatenate([data[ (k*num_chunks[1]) +  (j*num_chunks[2]) + i] for i in range(num_chunks[2])], axis=2) for j in range(num_chunks[1])], axis = 1) for k in range(num_chunks[0])], axis = 0)
                
    return x