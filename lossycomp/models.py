from tensorflow.keras.layers import Conv3D, Conv3DTranspose, ReLU, Activation, Input, Reshape, Flatten, Dense, PReLU, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
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

class ResAutoencoder:
  @staticmethod
  def build(time, latitude, longitude, channels):
    
    def ResBlock(x, num_filter):
      x_in = x
      x = Conv3D(num_filter, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
      x = ReLU()(x)
      x = Conv3D(num_filter, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
      x = x_in + x
      x = ReLU()(x)
      return x
    
    inputShape = (time, latitude, longitude, channels)
    
    # ----------------------------------------------------------------------------------------------------------#
    # Input to the encoder
    inputs = Input(shape=inputShape)
    x = inputs
    
    x = Conv3D(64, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
    x = PReLU()(x)
    
    x = Conv3D(128, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)

    x = ResBlock(x, 128)
    
    x = Conv3D(96, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    x = ResBlock(x, 96)
    
    x = Conv3D(64, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    x = ResBlock(x, 64)
    
    x = Conv3D(32, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    x = Conv3D(32, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)   
    x = PReLU()(x)
         
    volumeSize = K.int_shape(x)
    
    encoder = Model(inputs = inputs, outputs = x, name="Encoder")
        
    # ----------------------------------------------------------------------------------------------------------#
    # Input to the decoder
    latentInputs = Input(shape=volumeSize[1:])
    x = latentInputs

    x = Conv3D(128, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
    x = PReLU()(x)
    
    x = Conv3DTranspose(64, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)  # Stride 2.
    x = PReLU()(x)
    
    x = Conv3D(64, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)  
    x = PReLU()(x)
    
    x = ResBlock(x, 64)
    
    x = Conv3DTranspose(96, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    x = Conv3D(96, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)  
    x = PReLU()(x)
    
    x = ResBlock(x, 96)
    
    x = Conv3DTranspose(64, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    
    x = Conv3D(64, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)  
    x = PReLU()(x)
    
    x = ResBlock(x, 64)
    
    x = Conv3DTranspose(32, (5, 5, 5), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    x = Conv3D(1, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)  
    x = PReLU()(x)
    
    # build the decoder model
    decoder = Model(latentInputs, x, name="Decoder")
    
    # ----------------------------------------------------------------------------------------------------------#
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="Autoencoder")
        
    return (encoder, decoder, autoencoder)


class ResAutoencoder2:
  @staticmethod
  def build(time, latitude, longitude, channels):
    
    def ResBlock(x, num_filter):
      x_in = x
      x = Conv3D(num_filter, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
      x = ReLU()(x)
      x = Conv3D(num_filter, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
      x = x_in + x
      x = ReLU()(x)
      return x
    
    inputShape = (time, latitude, longitude, channels)
    
    # ----------------------------------------------------------------------------------------------------------#
    # Input to the encoder
    inputs = Input(shape=inputShape)
    x = inputs
    
    #x = Conv3D(64, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
    #x = PReLU()(x)
    
    x = Conv3D(128, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)

    #x = ResBlock(x, 128)
    
    x = Conv3D(96, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    #x = ResBlock(x, 96)
    
    x = Conv3D(64, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    #x = ResBlock(x, 64)
    
    x = Conv3D(32, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    #x = Conv3D(32, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)   
    #x = PReLU()(x)
         
    volumeSize = K.int_shape(x)
    
    encoder = Model(inputs = inputs, outputs = x, name="Encoder")
        
    # ----------------------------------------------------------------------------------------------------------#
    # Input to the decoder
    latentInputs = Input(shape=volumeSize[1:])
    x = latentInputs

    #x = Conv3D(128, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
    #x = PReLU()(x)
    
    x = Conv3DTranspose(64, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)  # Stride 2.
    x = PReLU()(x)
    
    x = Conv3D(64, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)  
    x = PReLU()(x)
    
    #x = ResBlock(x, 64)
    
    x = Conv3DTranspose(96, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    x = Conv3D(96, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)  
    x = PReLU()(x)
    
    #x = ResBlock(x, 96)
    
    x = Conv3DTranspose(64, (3, 3, 3), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    x = Conv3D(64, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)  
    x = PReLU()(x)
    
    #x = ResBlock(x, 64)
    
    x = Conv3DTranspose(32, (5, 5, 5), strides = (2, 2, 2), padding="same")(x)   # Stride 2.
    x = PReLU()(x)
    
    x = Conv3D(1, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)  
    x = PReLU()(x)
    
    # build the decoder model
    decoder = Model(latentInputs, x, name="Decoder")
    
    # ----------------------------------------------------------------------------------------------------------#
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="Autoencoder")
        
    return (encoder, decoder, autoencoder)


class ResAutoencoder3:
  @staticmethod
  def build(time, latitude, longitude, channels):

    inputShape = (time, latitude, longitude, channels)
    
    # ----------------------------------------------------------------------------------------------------------#
    # Input to the encoder
    inputs = Input(shape=inputShape)
    x = inputs

    
    x = Conv3D(10, (3, 3, 3), strides = (2, 2, 2), padding="same", activation = "relu")(x)   # Stride 2.

    x = Conv3D(20, (3, 3, 3), strides = (2, 2, 2), padding="same", activation = "relu")(x)   # Stride 2.
    
    x = Conv3D(20, (3, 3, 3), strides = (2, 2, 2), padding="same", activation = "relu")(x)   # Stride 2.
    
    x = Conv3D(20, (3, 3, 3), strides = (2, 2, 2), padding="same", activation = "sigmoid")(x)

    volumeSize = K.int_shape(x)
    
    encoder = Model(inputs = inputs, outputs = x, name="Encoder")
        
    # ----------------------------------------------------------------------------------------------------------#
    # Input to the decoder
    latentInputs = Input(shape=volumeSize[1:])
    x = latentInputs

    x = Conv3D(20, (3, 3, 3), strides = (1, 1, 1), padding="same", activation = "relu")(x)  
    x = Conv3DTranspose(20, (3, 3, 3), strides = (2, 2, 2), padding="same", activation = "relu")(x)  # Stride 2.
    
    x = Conv3D(20, (3, 3, 3), strides = (1, 1, 1), padding="same", activation = "relu")(x)  
    x = Conv3DTranspose(20, (3, 3, 3), strides = (2, 2, 2), padding="same", activation = "relu")(x)   # Stride 2.
    
    x = Conv3D(10, (3, 3, 3), strides = (1, 1, 1), padding="same", activation = "relu")(x)  
    x = Conv3DTranspose(10, (3, 3, 3), strides = (2, 2, 2), padding="same", activation = "relu")(x)   # Stride 2.
    
    x = Conv3D(10, (3, 3, 3), strides = (1, 1, 1), padding="same", activation = "relu")(x)
    x = Conv3DTranspose(10, (3, 3, 3), strides = (2, 2, 2), padding="same", activation = "relu")(x)   # Stride 2.
      
    x = Conv3D(1, (3, 3, 3), strides = (1, 1, 1), padding="same")(x)
    # build the decoder model
    decoder = Model(latentInputs, x, name="Decoder")
    
    # ----------------------------------------------------------------------------------------------------------#
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="Autoencoder")
        
    return (encoder, decoder, autoencoder)

    
        