"""Dataset classes and functions."""

import os
import random
import xarray as xr
import numpy as np
from tensorflow import keras
from lossycomp.constants import Region, REGIONS
from lossycomp.encodings import encode_lat, encode_lon

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, num_samples, leads, mean, std, batch_size=10, standardize = False, coords = False, soil = False, load=True):
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
        self.soil = soil
        self.check_inputs()
        
        if standardize:
            self.data = (self.data - self.mean) / self.std

        tim_diff, lat_diff, lon_diff, lev_diff, subset_shape = self.check_chunks()
        
        subset = self.data.isel(
            time = slice(None, tim_diff),
            level = slice(None, lev_diff),
            latitude = slice(None, lat_diff),
            longitude = slice(None, lon_diff),
        )
        
        
        self.subset_shape = subset_shape[0], subset_shape[1], subset_shape[2], subset_shape[3]
                
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
        
        whole_data = self.data.isel(**subset_selection)
        
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
            
            #xx, yy = np.meshgrid(lon, lat)
            
            coords_lat = np.concatenate([[xx]] * len(whole.time), axis=0)
            coords_lon = np.concatenate([[yy]] * len(whole.time), axis=0)
            coords_lat1 = np.concatenate([[xx2]] * len(whole.time), axis=0)
            coords_lon1 = np.concatenate([[yy2]] * len(whole.time), axis=0)
            coords_lat = np.expand_dims(coords_lat, axis=3)
            coords_lon = np.expand_dims(coords_lon, axis=3)
            coords_lat1 = np.expand_dims(coords_lat1, axis=3)
            coords_lon1 = np.expand_dims(coords_lon1, axis=3)
            soil_data = whole.coords['lsm'].values
            whole_data =  np.concatenate((whole_data, coords_lat, coords_lon, coords_lat1, coords_lon1, soil_data ),axis = 3)
            
        #if self.soil:
            #soil_data = whole.coords['lsm'].values
            #whole_data = np.concatenate((whole_data, soil_data),axis = 3)
            
        return whole_data
    
    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size : (i + 1) * self.batch_size]
        x = np.stack([self.calculateValues(i).data for i in idxs], axis = 0)
        x= np.array(x, dtype = np.float32)
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
        #random.seed(30)
        self.idxs = random.sample(range(0, self.subset_length), self.num_samples)
            
    def check_inputs(self):
        assert isinstance(self.data, xr.core.dataarray.DataArray)
        assert isinstance(self.leads, dict)
        assert all(var in self.leads for var in ['longitude', 'time', 'latitude', 'level'])
        
    def check_chunks(self):
        tim_diff = self.data.time.size - self.leads["time"]
        lat_diff = self.data.latitude.size - self.leads["latitude"] 
        lon_diff = self.data.longitude.size - self.leads["longitude"]
        lev_diff = self.data.level.size - self.leads["level"] 
        if (tim_diff < 0 or lat_diff < 0 or lon_diff < 0 or lev_diff < 0):
            raise ValueError("Chunk size can't be bigger than data size.")
            
        subset_shape = [tim_diff, lev_diff, lat_diff, lon_diff]
        
        if tim_diff == 0:
            tim_diff = len(self.data.time)
            subset_shape[0] = 1
        if lat_diff == 0:
            lat_diff = len(self.data.latitude)
            subset_shape[2] = 1
        if lon_diff == 0:
            lon_diff = len(self.data.longitude)
            subset_shape[3] = 1
        if lev_diff == 0:
            lev_diff = len(self.data.level)
            subset_shape[1] = 1
        return tim_diff, lat_diff, lon_diff, lev_diff, subset_shape
            
    
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
    mean = data.mean(('time', 'latitude', 'longitude')).compute() 
    std = data.std('time').mean(('latitude', 'longitude')).compute() 
 
    return (data, mean.data[0], std.data[0])
    
    
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
    Chunks data into (N, t, lat, lon, attr.)
    Args:
    ===================
    data: 4D - Numpy array (time, lat, lon, attr).
    size: (t, lat, lon, level) of one chunk.
    
    Returns N chunks with size (t, lat, long, attr.)
    """
    assert (data.shape[0] >= size[0] and data.shape[1] >= size[1] and data.shape[2] >= size[2]), 'Chunks size cannot be larger than the data size.'
    
    chunk_num = (int(np.ceil(data.shape[0]/size[0]) ), int(np.ceil(data.shape[1]/size[1])), int(np.ceil(data.shape[2] /size[2])))
    
    length = int(np.prod(chunk_num)) 
    
    x = np.stack([calculateValues(data, i, size, chunk_num) for i in range(length)], axis = 0)
    
    return x

def calculateValues(data, i, size, chunk_num):
    tix, latix, lonix = np.unravel_index(i, chunk_num)

    if (tix*size[0] + size[0]) > data.shape[0]:
        out = data[-size[0]:, latix*size[1]: latix*size[1] + size[1], lonix*size[2]: lonix*size[2] + size[2]]
        
        if ((latix*size[1] + size[1]) > data.shape[1]) and ((lonix*size[2] + size[2]) > data.shape[2]):
            out = data[-size[0]:, -size[1]:, -size[2]:]
        elif(latix*size[1] + size[1]) > data.shape[1]:
            out = data[-size[0]:, -size[1]: ,  lonix*size[2]: lonix*size[2] + size[2]]
        elif(lonix*size[2] + size[2]) > data.shape[2]:
            out = data[-size[0]:, latix*size[1]: latix*size[1] + size[1], -size[2]:]
        
    elif (latix*size[1] + size[1]) > data.shape[1]:
        out = data[tix*size[0]: tix*size[0] + size[0], -size[1]: , lonix*size[2]: lonix*size[2] + size[2]]
        if(lonix*size[2] + size[2]) > data.shape[2]:
            out = data[tix*size[0]: tix*size[0] + size[0], -size[1]: , -size[2]:]
        
    elif (lonix*size[2] + size[2]) > data.shape[2]:
        out = data[tix*size[0]: tix*size[0] + size[0], latix*size[1]: latix*size[1] + size[1], -size[2]:]

    else:
        out = data[tix*size[0]: tix*size[0] + size[0], latix*size[1]: latix*size[1] + size[1], lonix*size[2]: lonix*size[2] + size[2]]
        
    return out


def merge_data(data, size):
    """
    Merges data into (t, lat, lon, attr.)
    Args:
    =======================
    data: 5D - Numpy array.
    size: (t, lat, lon, attr.) size of the output data.
    
    Returns complete data (t, lat, long, attr.)
    """
    
    chunks_size = data.shape[1:4]
    num_chunks = (int(np.ceil(size[0]/chunks_size[0]) ), int(np.ceil(size[1]/chunks_size[1])), int(np.ceil(size[2] /chunks_size[2])))
    x = np.concatenate([ np.concatenate([np.concatenate(
        [ select_chunk(data,i,j,k,num_chunks, chunks_size, size)
        for i in range(num_chunks[2])], axis=2)for j in range(num_chunks[1])], axis = 1)
                         for k in range(num_chunks[0])], axis = 0)
    return x
     
def select_chunk(data, i, j, k, num_chunks, chunks_size, size):

    if ((i * chunks_size[2] +chunks_size[2]) > size[2]):
        mn = chunks_size[2] - ((i * chunks_size[2] + chunks_size[2])- size[2])
        out = data[(k*num_chunks[1]* num_chunks[2]) +  (j*num_chunks[2]) + i][:,:,-mn:]
        
        if ((j * chunks_size[1] + chunks_size[1])  > size[1]):
            mn = chunks_size[1] - ((j * chunks_size[1] + chunks_size[1]) - size[1])
            out = out[:,-mn:,:]
            
        if ((k * chunks_size[0]+ chunks_size[0]) >size[0]):
            mn = chunks_size[0] - ((k * chunks_size[0] + chunks_size[0]) - size[0])
            out = out[-mn:,:,:]  
        
            
    elif ((j * chunks_size[1] + chunks_size[1])  > size[1]):
        mn = chunks_size[1] - ((j * chunks_size[1] + chunks_size[1]) - size[1])
        out = data[(k*num_chunks[1]* num_chunks[2]) +  (j*num_chunks[2]) + i][:,-mn:,:]
        if ((k * chunks_size[0]+ chunks_size[0]) >size[0]):
            mn = chunks_size[0] - ((k * chunks_size[0] + chunks_size[0]) - size[0])
            out = out[-mn:,:,:]
            
    elif ((k * chunks_size[0]+ chunks_size[0]) > size[0]):
        mn = chunks_size[0] - ((k * chunks_size[0] + chunks_size[0]) - size[0])
        out = data[(k*num_chunks[1]* num_chunks[2]) +  (j*num_chunks[2]) + i][-mn:,:,:]


    else:
        out = data[(k*num_chunks[1]* num_chunks[2]) +  (j*num_chunks[2]) + i]
    return out