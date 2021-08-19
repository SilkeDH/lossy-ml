import numpy as np
import time
import xarray as xr
import subprocess
import os
import zfpy
import matplotlib.pyplot as plt
from lossycomp.compress_test import compress
from lossycomp.decompress_test import decompress
from lossycomp.plots import single_plot
from lossycomp.encodings import encode_lat, encode_lon
from lossycomp.constants import data_path

data_or = xr.open_mfdataset(data_path + 'data/ECMWF/1980/*/temperature.nc', combine='by_coords')
data_or = data_or.sel(time=slice('1980-5-1T00:00:00','1980-5-3T1:00:00'),level = slice(978,1000))
data_or = data_or.transpose('time', 'latitude', 'longitude', 'level')

extra = False
soil = True

abs_error = 0.01

if extra:
    # In case we use the extra information:
    lat = data_or['t'].coords['latitude'].values
    lon = data_or['t'].coords['longitude'].values

    lat_st = np.stack([encode_lat(x) for x in lat])
    lon_st = np.stack([encode_lon(x) for x in lon])

    lat1, lat2 = np.hsplit(lat_st, 2)
    lon1, lon2 = np.hsplit(lon_st, 2)

    xx, yy = np.meshgrid(lon1, lat1)
    xx2, yy2 = np.meshgrid(lon2, lat2)

    coords_lat = np.concatenate([[xx]] * len(data_or.time), axis=0)
    coords_lon = np.concatenate([[yy]] * len(data_or.time), axis=0)
    coords_lat1 = np.concatenate([[xx2]] * len(data_or.time), axis=0)
    coords_lon1 = np.concatenate([[yy2]] * len(data_or.time), axis=0)

    coords_lat = np.expand_dims(coords_lat, axis=3)
    coords_lon = np.expand_dims(coords_lon, axis=3)

    coords_lat1 = np.expand_dims(coords_lat1, axis=3)
    coords_lon1 = np.expand_dims(coords_lon1, axis=3)

    temp = data_or['t'].values
    data_or =  np.concatenate((temp, coords_lat, coords_lon, coords_lat1, coords_lon1 ),axis = 3)

    data_or.shape

if soil:
    data_soil = xr.open_mfdataset('/p/home/jusers/donayreholtz1/hdfml/MyProjects/PROJECT_haf/data/ECMWF/1980_single/*/land-sea-mask.nc', combine='by_coords')
    data_soil = data_soil.sel(time=slice('1980-5-1T00:00:00','1980-5-3T1:00:00'),longitude=slice(-180,180), latitude=slice(90,-90))
    data_soil = data_soil.transpose('time', 'latitude', 'longitude')
    data_soil = data_soil['lsm']
    data_soil = np.expand_dims(data_soil.values, axis=3)
    data_or =  np.concatenate((data_or['t'].values, data_soil ),axis = 3)

if not (soil or extra):
    data_or = data_or['t']
    data_or = data_or.values


print("Data shape:", data_or.shape)


data_r = np.expand_dims(data_or[:,:,:,0], axis=0)
data_r = np.expand_dims(data_r, axis=4)

print(data_r.shape)
plt.rcParams.update({'font.size': 16})
single_plot(data_r, 0, "Original Data", "Temperature(K)", data_r[0,0].min(), data_r[0,0].max())

np.save('original.npy', data_r)

compressed_data = compress(data_or, abs_error, extra_channels = extra, verbose = True, method='mask', mode = 'soil', convs = 4, hyp = 'final_models/model_1', enc_lat = 'fpzip')
decompressed_data = decompress(compressed_data[0], extra_channels = extra, verbose = False, method='mask', mode = 'soil', convs = 4, hyp = 'final_models/model_1', enc_lat = 'fpzip')

ae_decompressed = decompressed_data[1]
decompressed_data = decompressed_data[0]

def psnr(y,x):
    vrange = np.max(y) - np.min(y)
    psnr = 20 * np.log10(vrange - (10 * np.log10(np.mean((y-x)*(y-x)))))
    return psnr


psnr1 = psnr(data_or[0,:,:,0], decompressed_data[0,:,:,0])
print('PSNR model', psnr1)
print('Max error Model:', np.max(data_or[:,:,:,:]-decompressed_data[:,:,:,:]))

original = data_r
diff = (data_r-decompressed_data)

single_plot(np.expand_dims(decompressed_data, axis=0), 0, "Decompressed Data", "Temperature(K)", data_r[0,0].min(), data_r[0,0].max())
print('Decom shape', decompressed_data.shape)
np.save('decompressed_data.npy', decompressed_data)

#print(compressed_data[1].shape)
#print(diff.shape)

single_plot(np.expand_dims(ae_decompressed, axis=0), 0, "Decompressed AE Data", "Temperature(K)", data_r[0,0].min(), data_r[0,0].max())
print('AE shape', ae_decompressed.shape)
np.save('decompressed_data_AE.npy', ae_decompressed)

single_plot(diff, 0, "Difference", "Temperature(K)", diff[0,0].min(), diff[0,0].max(), cmap = plt.get_cmap('PuOr'))
print('Diff shape', diff.shape)
np.save('diff_' + str(abs_error) + '.npy',  diff)


diff2 = (data_r-ae_decompressed)
single_plot(diff2, 0, "Difference AE", "Temperature(K)", diff[0,0].min(), diff[0,0].max(), cmap = plt.get_cmap('PuOr'))
print('Diff AE shape', diff2.shape)
np.save('diff_ae.npy',  diff2)

print("")
print("Data Original")
print("=======")
print("Mean:", original.mean())
print("Standard Dev.:", original.std())
print("Max Val:", original.max())
print("Min Val:", original.min())
print("")
print("Decompressed")
print("=======")
print("Mean:", decompressed_data.mean())
print("Standard Dev.:", decompressed_data.std())
print("Max Val:", decompressed_data.max())
print("Min Val:", decompressed_data.min())
print("")
print("Error")
print("=======")
print("Mean:", diff.mean())
print("Standard Dev.:", diff.std())
print("Max error:", diff.max())
print("Min error:", diff.min())
print("")


plt.figure(figsize=(20,10))
plt.hist(diff.flatten(), 300)
plt.title('Error Histogram')
plt.xlabel('Error')
plt.ylabel('Count')

np.save('quant.npy', compressed_data[6])
np.save('mask.npy', compressed_data[7])
