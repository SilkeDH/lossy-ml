from tensorflow.keras.layers import Conv3D, Conv3DTranspose, ReLU, Activation, Input, Reshape, Flatten, Dense, PReLU, ReLU, Add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

class Autoencoder:
    @staticmethod
    def build(time, latitude, longitude, channels, filters=(10, 20, 20, 20), kernels = (4, 4, 4, 4), 
            strides = (2, 2, 2, 2), dropout = 0):
        inputShape = (time, latitude, longitude, channels)
        # Input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs
        # Loop over values
        for f, k, s in zip(filters, kernels, strides):
            x = Conv3D(f, (k, k, k), strides = (s, s, s), padding="same", activation = 'relu')(x)
        if dropout> 0:
            x = Dropout(dropout)(x)
        
        volumeSize = K.int_shape(x)
    
        encoder = Model(inputs = inputs, outputs = x, name="Encoder")
      
        # Input to the decoder
        latentInputs = Input(shape=volumeSize[1:])
        x = latentInputs

        kernels = np.flipud(kernels)
        filters = np.flipud(filters)
        strides = np.flipud(strides) 
        for f, k, s in zip(filters[1:], kernels[1:], strides[1:]):
            x = Conv3DTranspose(f, (k, k, k), strides = (s, s, s), padding="same", activation = 'relu')(x)
        if dropout> 0:
            x = Dropout(dropout)(x)
        
        x = Conv3DTranspose(filters = 1, kernel_size = (kernels[-1], kernels[-1] , kernels[-1]),
                        activation=None, strides = (strides[-1], strides[-1], strides[-1]) , padding="same")(x)  

        # build the decoder model
        decoder = Model(latentInputs, x, name="Decoder")
        autoencoder = Model(inputs, decoder(encoder(inputs)), name="Autoencoder")
        
        return (encoder, decoder, autoencoder)
    
    
class Autoencoder2:
    @staticmethod
    def build(time, latitude, longitude, channels, convs=4, filters=20, kernel = 3, lr= 0.001, res=0, l2 = 0.0005):
        inputShape = (time, latitude, longitude, channels)
        # Input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs
        
        # Residual Blocks
        if res > 0:
            x = Conv3D(filters, (kernel, kernel, kernel), strides = (1, 1, 1) , padding="same",  activation = 'relu')(x)
        for i in range(res):
            x = ResBlock(x, filters, l2)
        
         # Reducing conv blocks
        for i in range(convs):
            x = Conv3D(filters, (kernel, kernel, kernel), strides = (2, 2, 2) , padding="same", activation = 'relu')(x)
        volumeSize = K.int_shape(x)

        # Encoder
        encoder = Model(inputs = inputs, outputs = x, name="Encoder")

        # Input to the decoder
        latentInputs = Input(shape=volumeSize[1:])
        x = latentInputs

        # Deconvs blocks
        for i in range(convs):
            x = Conv3DTranspose(filters, (kernel, kernel, kernel), strides = (2, 2, 2) , padding="same", activation = 'relu')(x)
    
        # Residual blocks
        for i in range(res):
            x = ResBlock(x, filters, l2)
        
        # To one channel
        x = Conv3D(filters = 1, kernel_size = (kernel, kernel, kernel), activation = None, strides = 1 , padding="same")(x)  

        # build the decoder model
        decoder = Model(latentInputs, x, name="Decoder")
        autoencoder = Model(inputs, decoder(encoder(inputs)), name="Autoencoder")
        return (encoder, decoder, autoencoder)

    
def ResBlock(x, num_filter, hp6): # res v1
    x_in = x
    x = Conv3D(num_filter, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
    x = ReLU()(x)
    x = Conv3D(num_filter, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
    x = Add()([x, x_in])
    x = ReLU()(x)
    return x