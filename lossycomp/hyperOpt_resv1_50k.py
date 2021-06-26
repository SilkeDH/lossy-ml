from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, ReLU, Activation, Input, Reshape, Flatten, Dense, PReLU, ReLU, Add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import dask
from collections import OrderedDict
from lossycomp.dataLoader import DataGenerator, data_preprocessing, split_data
from lossycomp.utils import lr_log_reduction, correlation_5, calculate_MAE_5, mean_squared_error_5

class Autoencoder_2(HyperModel):
    def __init__(self, time, latitude, longitude, channels, strides = 2):
        self.time = time
        self.latitude = latitude
        self.longitude = longitude
        self.channels = channels
        self.strides = strides
    
    def build(self, hp):
        
        def ResBlock(x, num_filter): # res v1
            x_in = x
            x = Conv3D(num_filter, (3, 3, 3), strides = (1, 1, 1), padding="same", kernel_regularizer= tf.keras.regularizers.l2(hp6))(x)
            x = ReLU()(x)
            x = Conv3D(num_filter, (3, 3, 3), strides = (1, 1, 1), padding="same", kernel_regularizer= tf.keras.regularizers.l2(hp6))(x)
            x = Add()([x, x_in])
            x = ReLU()(x)
            return x
    
        inputShape = (self.time, self.latitude, self.longitude, self.channels)
        # Input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs
        
		# Define hyperparameters
        hp1 = hp.Int('convs', min_value=4, max_value= 6, step=1)  # Hyp 1: Number of convolutions [4,5,6]
        
        hp2 = hp.Int('filter', min_value=20, max_value= 64, step=1) # Hyp 2: Number of filters.
        
        hp3 = hp.Int('kernel_siz', min_value=3, max_value= 7, step=1) #Hyp 3: Kernel size [3,4,5,6,7]
		
        hp4 = hp.Float('Learning_rate', min_value = 0.0001, max_value = 0.01, sampling = 'log') # Hyp 4: Learning rate (0.01, 0.0001)

        hp5 = hp.Int('num_res_blocks', 0, 3) # Hyp 5: Number of residual blocks [0,1,2,3]
        
        hp6 = hp.Float('l2_reg', min_value = 0.00005 , max_value = 0.5, sampling = 'log' ) # Hyp 6: Number of residual blocks [0,1,2,3]
        
        x = Conv3D(hp2, (hp3, hp3, hp3), strides = (1,1, 1) , padding="same", kernel_regularizer= tf.keras.regularizers.l2(hp6), activation = 'relu')(x)

		# Residual Blocks
        for i in range(hp5):
            x = ResBlock(x, hp2)
        
        # Reducing conv blocks
        for i in range(hp1):
            x = Conv3D(hp2, (hp3, hp3, hp3), strides = (self.strides,self.strides,self.strides) , padding="same", kernel_regularizer= tf.keras.regularizers.l2(hp6), activation = 'relu')(x)
        
        volumeSize = K.int_shape(x)
    
        # Encoder
        encoder = Model(inputs = inputs, outputs = x, name="Encoder")
        
        # Input to the decoder
        latentInputs = Input(shape=volumeSize[1:])
        
        x = latentInputs
        
        # Deconvs blocks
        for i in range(hp1):
            x = Conv3DTranspose(hp2, (hp3, hp3, hp3), strides = (self.strides,self.strides,self.strides) , padding="same", kernel_regularizer= tf.keras.regularizers.l2(hp6), activation = 'relu')(x)
        
        # Residual blocks
        for i in range(hp5):
            x = ResBlock(x, hp2)
        
        # To one channel
        x = Conv3D(filters = 1, kernel_size = (hp3, hp3, hp3), activation = None, strides = 1 , padding="same", kernel_regularizer= tf.keras.regularizers.l2(hp6))(x)  

        # build the decoder model
        decoder = Model(latentInputs, x, name="Decoder")
        autoencoder = Model(inputs, decoder(encoder(inputs)), name="Autoencoder")
            
        autoencoder.compile(optimizer=keras.optimizers.Adam(hp4),
                      loss=mean_squared_error_5, 
                      metrics=[ tf.keras.losses.MeanSquaredError(), 'MAE']) 
        print(autoencoder.summary())
        return autoencoder
        
        
hypermodel = Autoencoder_2(64, 64, 64, 5) #The more convs with stride 2, the bigger the data chunks.

tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=50,
    directory='../results/5_channel_hyperOpt_resv1_400k',
    project_name='AE_model')

tuner.search_space_summary()

dask.config.set(**{'array.slicing.split_large_chunks': False})

file = '/lsdf/kit/scc/projects/abcde/1979/*/ERA5.pl.temperature.nc'
region = "globe"
var = OrderedDict({'t': 1000})

z, mean, std = data_preprocessing(file, var, region)

train, test = split_data(z, 0.70)

leads = dict(time = 64, longitude=64, latitude=64, level=1)

dg_train = DataGenerator(train, 50000, leads, batch_size=100, load=True, mean= mean, std=std, coords = True, standardize = True) 
dg_test = DataGenerator(test, 5000, leads, batch_size=100, load=True, mean= mean, std=std, coords = True, standardize = True)

tuner.search(dg_train, epochs=100,  validation_data=dg_test, validation_freq = 5)