"""Dataset classes and functions."""

import os
import random
import xarray as xr
import numpy as np
from tensorflow import keras
from lossycomp.constants import Region, REGIONS


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, num_samples, leads, mean, std, batch_size=10, load=True):
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
            std: Standart deviation from dataset.
        """
        self.data = data
        self.num_samples = num_samples
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.leads = leads
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
        return whole
    
    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size : (i + 1) * self.batch_size]
        x = np.stack([self.calculateValues(i).data for i in idxs], axis = 0)
        return x, x

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
    