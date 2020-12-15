"""4DCNN Operations"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

def conv4d_stacked(input, filter,
                   strides=[1, 1, 1, 1, 1, 1],
                   padding='SAME',
                   dilation_rate=None,
                   stack_axis=None,
                   stack_nested=False,
                   ):
    # heuristically choose stack_axis
    if stack_axis is None:
        if dilation_rate is None:
            dil_array = np.ones(4)
        else:
            dil_array = np.asarray(dilation_rate)
        outputsizes = (np.asarray(input.get_shape().as_list()[1:5]) /
                       np.asarray(strides[1:5]))
        outputsizes -= dil_array*(
                            np.asarray(filter.get_shape().as_list()[:4])-1)
        stack_axis = np.argmin(outputsizes)+1

    if dilation_rate is not None:
        dilation_along_stack_axis = dilation_rate[stack_axis-1]
    else:
        dilation_along_stack_axis = 1

    tensors_t = tf.unstack(input, axis=stack_axis)
    kernel_t = tf.unstack(filter, axis=stack_axis-1)

    noOfInChannels = input.get_shape().as_list()[-1]
    len_ts = filter.get_shape().as_list()[stack_axis-1]
    size_of_t_dim = input.get_shape().as_list()[stack_axis]

    if len_ts % 2 == 1:
        # uneven filter size: same size to left and right
        filter_l = int(len_ts/2)
        filter_r = int(len_ts/2)
    else:
        # even filter size: one more to right
        filter_l = int(len_ts/2) - 1
        filter_r = int(len_ts/2)

    # The start index is important for strides and dilation
    # The strides start with the first element
    # that works and is VALID:
    start_index = 0
    if padding == 'VALID':
        for i in range(size_of_t_dim):
            if len(range(max(i - dilation_along_stack_axis*filter_l, 0),
                         min(i + dilation_along_stack_axis*filter_r+1,
                             size_of_t_dim),
                         dilation_along_stack_axis)) == len_ts:
                # we found the first index that doesn't need padding
                break
        start_index = i

    # loop over all t_j in t
    result_t = []
    for i in range(start_index, size_of_t_dim, strides[stack_axis]):

        kernel_patch = []
        input_patch = []
        tensors_t_convoluted = []

        if padding == 'VALID':
            # Get indices t_s
            indices_t_s = range(max(i - dilation_along_stack_axis*filter_l, 0),
                                min(i + dilation_along_stack_axis*filter_r+1,
                                    size_of_t_dim),
                                dilation_along_stack_axis)
            # check if Padding = 'VALID'
            if len(indices_t_s) == len_ts:

                # sum over all remaining index_t_i in indices_t_s
                for j, index_t_i in enumerate(indices_t_s):
                    if not stack_nested:
                        kernel_patch.append(kernel_t[j])
                        input_patch.append(tensors_t[index_t_i])
                    else:
                        if dilation_rate is not None:
                            tensors_t_convoluted.append(
                                tf.nn.convolution(
                                    tensors_t[index_t_i],
                                    kernel_t[j],
                                    strides=(strides[1:stack_axis+1]
                                             + strides[stack_axis:5]),
                                    padding=padding,
                                    dilations=(
                                            dilation_rate[:stack_axis-1]
                                            + dilation_rate[stack_axis:]))
                                )
                        else:
                            tensors_t_convoluted.append(
                                tf.nn.conv3d(input=tensors_t[index_t_i],
                                             filters=kernel_t[j],
                                             strides=(strides[:stack_axis] +
                                                      strides[stack_axis+1:]),
                                             padding=padding)
                                )
                if stack_nested:
                    sum_tensors_t_s = tf.add_n(tensors_t_convoluted)
                    # put together
                    result_t.append(sum_tensors_t_s)

        elif padding == 'SAME':

            # Get indices t_s
            indices_t_s = range(i - dilation_along_stack_axis*filter_l,
                                (i + 1) + dilation_along_stack_axis*filter_r,
                                dilation_along_stack_axis)

            for kernel_j, j in enumerate(indices_t_s):
                # we can just leave out the invalid t coordinates
                # since they will be padded with 0's and therfore
                # don't contribute to the sum

                if 0 <= j < size_of_t_dim:
                    if not stack_nested:
                        kernel_patch.append(kernel_t[kernel_j])
                        input_patch.append(tensors_t[j])
                    else:
                        if dilation_rate is not None:
                            tensors_t_convoluted.append(
                                tf.nn.convolution(
                                    tensors_t[j],
                                    kernel_t[kernel_j],
                                    strides=(strides[1:stack_axis+1] +
                                             strides[stack_axis:5]),
                                    padding=padding,
                                    dilations=(
                                        dilation_rate[:stack_axis-1] +
                                        dilation_rate[stack_axis:]))
                                )
                        else:
                            tensors_t_convoluted.append(
                                tf.nn.conv3d(input=tensors_t[j],
                                             filters=kernel_t[kernel_j],
                                             strides=(strides[:stack_axis] +
                                                      strides[stack_axis+1:]),
                                             padding=padding)
                                        )
            if stack_nested:
                sum_tensors_t_s = tf.add_n(tensors_t_convoluted)
                # put together
                result_t.append(sum_tensors_t_s)

        if not stack_nested:
            if kernel_patch:
                kernel_patch = tf.concat(kernel_patch, axis=3)
                input_patch = tf.concat(input_patch, axis=4)
                if dilation_rate is not None:
                    result_patch = tf.nn.convolution(
                        input_patch,
                        kernel_patch,
                        strides=(strides[1:stack_axis] +
                                 strides[stack_axis+1:5]),
                        padding=padding,
                        dilations=(dilation_rate[:stack_axis-1] +
                                   dilation_rate[stack_axis:]))
                else:
                    result_patch = tf.nn.conv3d(
                                            input=input_patch,
                                            filters=kernel_patch,
                                            strides=(strides[:stack_axis] +
                                                     strides[stack_axis+1:]),
                                            padding=padding)
                result_t.append(result_patch)
    # stack together
    return tf.stack(result_t, axis=stack_axis)


def test_4dcnn():
    data = tf.constant(tf.ones((1, 2, 2, 2, 2, 1)),
                           dtype=tf.float32)
    kernel = tf.constant(tf.ones((2, 2, 2, 2, 1, 1)),
                             dtype=tf.float32)

    result = conv4d_stacked(input=data, filter=kernel)
    return result

class Conv4D(layers.Layer):
    def __init__(self,
                 input_shape,
                 filter_size,
                 num_filters,
                 pooling_type=None,
                 pooling_strides=None,
                 pooling_ksize=None,
                 pooling_padding='SAME',
                 use_dropout=False,
                 activation='elu',
                 strides=None,
                 padding='SAME',
                 use_batch_normalisation=False,
                 dilation_rate=None,
                 use_residual=False,
                 method='convolution',
                 var_list=None,
                 weights=None,
                 biases=None,
                 trafo=None,
                 hex_num_rotations=1,
                 hex_azimuth=None,
                 hex_zero_out=False,
                 float_precision=tf.float32,
                 name=None):
        
        super(Conv4D, self).__init__()   
        
        self._padding= padding
        self._dilation_rate = dilation_rate
        self.filter_size = filter_size
        self._activation = activation

        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()
            
        # check dimension of input
        num_dims = len(input_shape) - 2
    
        # 4D convolution
        #if self.pooling_strides is None:
        #    self.pooling_strides = [1, 2, 2, 2, 2, 1]
        #if self.pooling_ksize is None:
        #    self.pooling_ksize = [1, 2, 2, 2, 2, 1]
        if strides is None:
            self._strides = [1, 1, 1, 1, 1, 1]
        
        num_input_channels = input_shape[-1]
        shape = list(filter_size) + [num_input_channels, num_filters]
        
        if weights is None:
            # weights = new_kernel_weights(shape=shape)
             self._weights = new_weights(shape=shape, float_precision=float_precision)

            # Create new biases, one for each filter.
        if biases is None:
            self._biases = new_biases(length=num_filters, float_precision=float_precision)

        return 

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        
        layer = conv4d_stacked(input=inputs,filter=self._weights,strides=self._strides,padding=self._padding,
                                    dilation_rate=self._dilation_rate)
        if self._biases is not None:
            layer = (layer + self._biases) / np.sqrt(2.)
        return layer


def new_weights(shape, stddev=1.0, name="weights",
                float_precision=tf.float32):
    """Helper-function to create new weights
    Args
    ========
    shape : list of int
        The desired shape.
    stddev : float, optional
        The initial values are sampled from a truncated gaussian with this
        std. deviation.
    name : str, optional
        The name of the tensor.
    float_precision : tf.dtype, optional
        The tensorflow dtype describing the float precision to use.
    Returns
    =========
    tf.Tensor
        A tensor with the weights.
    """
    return tf.Variable(tf.random.truncated_normal(
                            shape, stddev=stddev, dtype=float_precision),
                       name=name,
                       dtype=float_precision)

def new_biases(length, stddev=1.0, name='biases',
               float_precision=tf.float32):
    """Get new biases.
    Parameters
    ==========
    length : int
        Number of biases to get.
    stddev : float, optional
        The initial values are sampled from a truncated gaussian with this
        std. deviation.
    name : str, optional
        The name of the tensor.
    float_precision : tf.dtype, optional
        The tensorflow dtype describing the float precision to use.
    Returns
    ===========
    tf.Tensor
        A tensor with the biases.
    """
    return tf.Variable(tf.random.truncated_normal(shape=[length],
                                                  stddev=stddev,
                                                  dtype=float_precision),
                       name=name, dtype=float_precision)


""" Usage:
data = tf.constant(tf.ones((789, 2, 2, 2, 2, 1)),
                           dtype=tf.float32)
kernel = tf.constant(tf.ones((2, 2, 2, 2, 1, 1)),
                             dtype=tf.float32)

model = keras.Sequential(
    [
        Conv4D(input_shape=data.get_shape(), filter_size = [3, 3, 3, 3] , num_filters = 3),
    ]
)
# Compile the model
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.RMSprop(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)



# Train the model
model.fit(data, data, batch_size=32, epochs=100)
model.summary()
"""