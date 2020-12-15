# Lossy Compresion Algorithm for Climate Data

<p align=center>
  <img src="figures/planet_erde.png" />
</p>

## Dataset

We are working with the "ERA5 hourly data on pressure levels from 1979 to present" and the "ERA5 hourly data on single levels from 1979 to present" datasets provided by https://cds.climate.copernicus.eu.

### ERA5 data

- Each file has four dimensions and one attribute.
- These dimensions are called `time`, `level`, `latitude`, `longitude`.
- The `time.dtype` dimension is `np.datetime64`.
- The `level` dimension is given in `Pascal`.
- The `longitude` dimension is **ascending** and within the range of `[-180;180]`.
- The `latitude` dimension is **ascending** and within the range of `[-90;90]`.
- Attributes like `temperature`, `humidity`, etc.
- Each datafile represents a single month for a single year for a single attribute.
- The resolution of each datafile ist `t x 37 x 720 x 1440`.
- `t` is time in hours for each month. Roughly `24 x 30 = ~720`.

## Model

The compression algorithm is an Autoencoder based one. An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data. Along with the reduction side, a reconstructing side is learnt, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input, hence its name.

## Performance

The performance of the presented compression algorithm will be compared with zfp (https://computing.llnl.gov/projects/zfp). 
zfp is a BSD licensed open-source library for compressed floating-point arrays that support high throughput read and write random access.

## References

Datasets
- https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
- https://github.com/ucyo/ml-atmonn/blob/master/atmonn/datasets.py

Tensorflow
- https://www.tensorflow.org/tutorials/generative/cvae
- https://www.tensorflow.org/api_docs

4D Convolutions
- https://github.com/IceCubeOpenSource/TFScripts/blob/master/tfscripts/layers.py
