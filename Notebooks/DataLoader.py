import xarray as xr
import numpy as np
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, chunk_time, batch_size=32, shuffle=True, load=True, mean=None, std=None):
        """
        Data generator.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """
        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_time = chunk_time

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'latitude', 'longitude', 'level')
        self.mean = self.data.mean(('time', 'latitude', 'longitude')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('latitude', 'longitude')).compute() if std is None else std
        
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.values.shape[0]

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(np.ceil(self.n_samples / self.chunk_time) / self.batch_size ))


    def __getitem__(self, i):
        'Generate one batch of data'
        x = []
        for batch in range(self.batch_size):
            idxs = self.idxs[(i * self.batch_size * self.chunk_time) + batch : ((batch + 1) + (i * self.batch_size * self.chunk_time)) + (self.chunk_time-1)] 
            x.append(np.expand_dims(self.data.isel(time=idxs).values, axis=0))
        x = np.concatenate(x, axis =0)
        return x, x

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)