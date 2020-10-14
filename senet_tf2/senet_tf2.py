#!/usr/bin/env python
# coding: utf-8

# senet_tf2.py

"""
Please remember that it is the TensorFlow realization with image_data_foramt = 'channels_last'. If
the env of Keras is 'channels_first', please change it  according to the TensorFlow convention.  
Grouped Convolution Layer is implemented as a Slice, Conv2D and Concatenate Split filters to groups. 

# Set up the GPU memory size to avoid the out-of-memory error if the GPU setting has a problem 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

Make the necessary changes to adapt to the environment of TensorFlow 2.3, Keras 2.4.3, CUDA Toolkit 11.0, 
cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to replace the deprecated code. The 
script has many modifications on the foundation of is ResNet50 by Francios Chollet, qubvel and many other 
published results. I would like to thank all of them for the contributions. 

Environment: 
Ubuntu 18.04 
TensorFlow 2.3
Keras 2.4.3
CUDA Toolkit 11.0, 
cuDNN 8.0.1
CUDA 450.57. 

Reference
SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
https://arxiv.org/abs/1602.07360
"""


import collections
import numpy as np
import warnings
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Add, Input, Conv2D, Dropout, Dense, Activation, Flatten, Lambda, Multiply, \
    Concatenate, BatchNormalization, ZeroPadding2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import get_source_inputs
from imagenet_utils import _obtain_input_shape


# Set up the GPU to avoid the runtime error: Could not create cuDNN handle...
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


BASE_WEIGHTS_PATH = ('https://github.com/qubvel/classification_models/releases/download/0.0.1/')

WEIGHTS_HASHES = {
    'seresnet50':  ('ff0ce1ed5accaad05d113ecef2d29149',
                    '043777781b0d5ca756474d60bf115ef1'),
    'seresnet101': ('5c31adee48c82a66a32dee3d442f5be8',
                    '1c373b0c196918713da86951d1239007'),
    'seresnet152': ('96fc14e3a939d4627b0174a0e80c7371',
                    'f58d4c1a511c7445ab9a2c2b83ee4e7b'),
    'seresnext50': ('5310dcd58ed573aecdab99f8df1121d5',
                    'b0f23d2e1cd406d67335fb92d85cc279'),
    'seresnext101':('be5b26b697a0f7f11efaa1bb6272fc84',
                    'e48708cbe40071cc3356016c37f6c9c7'),
    'senet154':    ('c8eac0e1940ea4d8a2e0b2eb0cdf4e75',
                    'd854ff2cd7e6a87b05a8124cd283e0f2')
}


# Define the ModelParams(called in the middle of the script)
ModelParams = collections.namedtuple(
    'ModelParams',
    ['name', 'repetitions', 'residual_block', 'groups',
     'reduction', 'init_filters', 'input_3x3', 'dropout']
)

# ----------------------------------------------------------------------------------------------------------
# Give the helper functions
# ----------------------------------------------------------------------------------------------------------

def get_bn_params(**params):
    axis = -1 if K.image_data_format() == 'channels_last' else 1
    default_bn_params = {'axis': axis, 'epsilon': 9.99e-06,}
    default_bn_params.update(params)

    return default_bn_params


def get_num_channels(tensor):
    channels_axis = -1 if K.image_data_format() == 'channels_last' else 1
    return K.int_shape(tensor)[channels_axis]


def slice_tensor(x, start, stop, axis):
    if axis == -1:
        return x[:, :, :, start:stop]
    elif axis == 1:
        return x[:, start:stop, :, :]
    else:
        raise ValueError("Slice axis should be in (1,-1), got {}.".format(axis))


# ----------------------------------------------------------------------------------------------------------
# Pre-define GroupConv2D and ChannelSE 
# ----------------------------------------------------------------------------------------------------------

def GroupConv2D(filters, kernel_size, strides=(1,1), groups=32, kernel_initializer='he_uniform',
                use_bias=True, activation='linear', padding='valid', **kwargs):
    """
    Args:
        filters: Integer, the dimensionality of the output space
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer
        groups: Integer, number of groups to split input filters to.
        kernel_initializer: Regularizer function applied to the kernel weights matrix.
        use_bias: Boolean, whether the layer uses a bias vector.
        activation: Activation function to use,  inear activation: a(x) = x
        padding: one of "valid" or "same" (case-insensitive).
    Input shape:
        4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".
    Output shape:
        4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last".
        rows and cols values might have changed due to padding.
    """
    slice_axis = -1 if K.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        in_channel = int(K.int_shape(input_tensor)[-1] // groups)  # Input grouped channels
        out_channel = int(filters // groups)  # Output grouped channels

        blocks = []

        for c in range(groups):
            slice_arguments = {
                'start': c * in_channel,
                'stop': (c + 1) * in_channel,
                'axis': slice_axis,
            }
            x = Lambda(slice_tensor, arguments=slice_arguments)(input_tensor)
            x = Conv2D(out_channel, kernel_size,strides=strides, kernel_initializer=kernel_initializer,
                       use_bias=use_bias, activation=activation, padding=padding)(x)
            blocks.append(x)

        x = Concatenate(axis=slice_axis)(blocks)

        return x

    return layer


def expand_dims(x, channels_axis):
    if channels_axis == -1:
        return x[:, None, None, :]
    elif channels_axis == 1:
        return x[:, :, None, None]
    else:
        raise ValueError("Slice axis should be in (1,-1), got {}.".format(channels_axis))


def ChannelSE(reduction=16, **kwargs):
    """
    Args:
        reduction: channels squeeze factor
    Return
        layer
    """
    channels_axis = -1 if K.image_data_format() == 'channels_last' else 1

    def layer(input_tensor):
        # Get the number of channels/filters
        channels = K.int_shape(input_tensor)[channels_axis]

        x = input_tensor

        # Squeeze and excitation block
        x = GlobalAveragePooling2D()(x)
        x = Lambda(expand_dims, arguments={'channels_axis': channels_axis})(x)
        x = Conv2D(channels // reduction, kernel_size=(1,1), kernel_initializer='he_uniform')(x)
        x = Activation('relu')(x)
        x = Conv2D(channels, kernel_size=(1,1), kernel_initializer='he_uniform')(x)
        x = Activation('sigmoid')(x)

        # Apply the attention
        x = Multiply()([input_tensor, x])

        return x

    return layer

# ----------------------------------------------------------------------------------------------------------
# Residual blocks: SEResNetBottleneck and SEResNeXtBottleneck
# ----------------------------------------------------------------------------------------------------------

def SEResNetBottleneck(filters, reduction=16, strides=1, **kwargs):
    bn_params = get_bn_params()

    def layer(input_tensor):
        x = input_tensor
        residual = input_tensor

        # Give the bottleneck
        x = Conv2D(filters // 4, kernel_size=(1,1), kernel_initializer='he_uniform', 
                   strides=strides, use_bias=False)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation('relu')(x)

        x = ZeroPadding2D(1)(x)
        x = Conv2D(filters // 4, kernel_size=(3,3), kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size=(1,1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(**bn_params)(x)

        # If filter # or spatial dimensions changed make same manipulations with residual connection
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if strides != 1 or x_channels != r_channels:
            residual = Conv2D(x_channels, kernel_size=(1,1), strides=strides, 
                              kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = BatchNormalization(**bn_params)(residual)

        # Apply the attention module
        x = ChannelSE(reduction=reduction, **kwargs)(x)

        # Add the residual connection
        x = Add()([x, residual])

        x = Activation('relu')(x)

        return x

    return layer


def SEResNeXtBottleneck(filters, reduction=16, strides=1, groups=32, base_width=4, **kwargs):
    bn_params = get_bn_params()

    def layer(input_tensor):
        x = input_tensor
        residual = input_tensor

        width = (filters // 4) * base_width * groups // 64

        # bottleneck
        x = Conv2D(width, kernel_size=(1,1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation('relu')(x)

        x = ZeroPadding2D(1)(x)
        x = GroupConv2D(width, kernel_size=(3,3), strides=strides, groups=groups,
                        kernel_initializer='he_uniform', use_bias=False, **kwargs)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size=(1,1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(**bn_params)(x)

        # As same as the above
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if strides != 1 or x_channels != r_channels:
            residual = Conv2D(x_channels, kernel_size=(1,1), strides=strides, 
                              kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = BatchNormalization(**bn_params)(residual)

        # Apply attention module
        x = ChannelSE(reduction=reduction, **kwargs)(x)

        # Add residual connection
        x = Add()([x, residual])

        x = Activation('relu')(x)

        return x

    return layer


def SEBottleneck(filters, reduction=16, strides=1, groups=64, is_first=False, **kwargs):
    bn_params = get_bn_params()

    if is_first:
        downsample_kernel_size = (1,1)
        padding = False
    else:
        downsample_kernel_size = (3,3)
        padding = True

    def layer(input_tensor):

        x = input_tensor
        residual = input_tensor

        # Bottleneck
        x = Conv2D(filters // 2, kernel_size=(1,1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation('relu')(x)

        x = ZeroPadding2D(1)(x)
        x = GroupConv2D(filters, kernel_size=(3,3), strides=strides, groups=groups,
                        kernel_initializer='he_uniform', use_bias=False, **kwargs)(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size=(1,1), kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(**bn_params)(x)

        # As same as the above
        x_channels = get_num_channels(x)
        r_channels = get_num_channels(residual)

        if strides != 1 or x_channels != r_channels:
            if padding:
                residual = ZeroPadding2D(1)(residual)
            residual = Conv2D(x_channels, downsample_kernel_size, strides=strides, 
                              kernel_initializer='he_uniform', use_bias=False)(residual)
            residual = BatchNormalization(**bn_params)(residual)

        x = ChannelSE(reduction=reduction, **kwargs)(x)

        x = Add()([x, residual])

        x = Activation('relu')(x)

        return x

    return layer


# ----------------------------------------------------------------------------------------------------------
#  Define the SeNet builder
# ----------------------------------------------------------------------------------------------------------

# Please remember to modify model_params
def SENet(model_params, model_name, input_tensor=None, input_shape=None, include_top=True, 
          num_classes=1000, weights='imagenet', **kwargs):
    # Instantiates the ResNet, SEResNet architecture.
    """
    Args:
        include_top: whether to include the FC layer at the top of the network.
        weights: `None` (random initialization), 'imagenet' or the path to any weights.
        input_tensor: optional Keras tensor (output of `layers.Input()`)
        input_shape: tuple, only to be specified if `include_top` is False.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights` or invalid input shape.
    """
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    residual_block = model_params.residual_block
    init_filters = model_params.init_filters
    bn_params = get_bn_params()

    # Give the input
    if input_tensor is None:
        input = Input(shape=input_shape, name='input')
    else:
        if not K.is_keras_tensor(input_tensor):
            input = Input(tensor=input_tensor, shape=input_shape)
        else:
            input = input_tensor

    x = input

    if model_params.input_3x3:

        x = ZeroPadding2D(1)(x)
        x = Conv2D(init_filters, kernel_size=(3,3), strides=2, use_bias=False, 
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation('relu')(x)

        x = ZeroPadding2D(1)(x)
        x = Conv2D(init_filters, kernel_size=(3,3), use_bias=False,  
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation('relu')(x)

        x = ZeroPadding2D(1)(x)
        x = Conv2D(init_filters * 2, kernel_size=(3,3), use_bias=False, 
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation('relu')(x)

    else:
        x = ZeroPadding2D(3)(x)
        x = Conv2D(init_filters, kernel_size=(7,7), strides=2, use_bias=False, 
                   kernel_initializer='he_uniform')(x)
        x = BatchNormalization(**bn_params)(x)
        x = Activation('relu')(x)

    x = ZeroPadding2D(1)(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)

    # Give the body of the ResNet
    filters = model_params.init_filters * 2
    for i, stage in enumerate(model_params.repetitions):
        # Increase number of filters with each stage
        filters *= 2
        for j in range(stage):
            # Decrease spatial dimensions for each stage (except the first--maxpool)
            if i == 0 and j == 0:
                x = residual_block(filters, reduction=model_params.reduction, strides=1, 
                                   groups=model_params.groups, is_first=True, **kwargs)(x)
            elif i != 0 and j == 0:
                x = residual_block(filters, reduction=model_params.reduction, strides=2, 
                                   groups=model_params.groups, **kwargs)(x)
            else:
                x = residual_block(filters, reduction=model_params.reduction, strides=1, 
                                   groups=model_params.groups, **kwargs)(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        if model_params.dropout is not None:
            x = Dropout(model_params.dropout)(x)
        x = Dense(num_classes)(x)
        x = Activation('softmax', name='output')(x)

    # Ensure the model considers any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = input

    model = Model(inputs, x, name=model_name)

    # Load weights
    if weights == 'imagenet':
        if include_top:
            file_suff = '_imagenet_1000.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_suff = '_imagenet_1000_no_top.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        file_name = model_name + file_suff
        weights_path = get_file(file_name, BASE_WEIGHTS_PATH + file_name,
                                cache_subdir='models', file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


# ----------------------------------------------------------------------------------------------------------
# Give the arguments of the SE Residual Models
# ----------------------------------------------------------------------------------------------------------

MODELS_PARAMS = {
    'SEResNet50': ModelParams(
        'SEResNet50', repetitions=(3, 4, 6, 3), residual_block=SEResNetBottleneck,
        groups=1, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    ),

    'SEResNet101': ModelParams(
        'SEResNet101', repetitions=(3, 4, 23, 3), residual_block=SEResNetBottleneck,
        groups=1, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    ),

    'SEResNet152': ModelParams(
        'SEResNet152', repetitions=(3, 8, 36, 3), residual_block=SEResNetBottleneck,
        groups=1, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    ),

    'SEResNeXt50': ModelParams(
        'SEResneXt50', repetitions=(3, 4, 6, 3), residual_block=SEResNeXtBottleneck,
        groups=32, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    ),

    'SEResNeXt101': ModelParams(
        'SEResNeXt101', repetitions=(3, 4, 23, 3), residual_block=SEResNeXtBottleneck,
        groups=32, reduction=16, init_filters=64, input_3x3=False, dropout=None,
    ),

    'SENet154': ModelParams(
        'SENet154', repetitions=(3, 8, 36, 3), residual_block=SEBottleneck,
        groups=64, reduction=16, init_filters=64, input_3x3=True, dropout=0.2,
    ),
}


# ----------------------------------------------------------------------------------------------------------
# Call the function of SENet for specific models 
# ----------------------------------------------------------------------------------------------------------
# Please see there is no model_name but MODELS_PARAMS
def SEResNet50(input_shape=None, model_name=None, input_tensor=None, weights=None, num_classes=1000, include_top=True, **kwargs):
    return SENet(MODELS_PARAMS['SEResNet50'], model_name='seresnet50', input_shape=input_shape, input_tensor=input_tensor,
                 include_top=include_top, num_classes=num_classes, weights=weights, **kwargs)


def SEResNet101(input_shape=None, input_tensor=None, weights=None, num_classes=1000, include_top=True, **kwargs):
    return SENet(MODELS_PARAMS['SEResNet101'], model_name='seresnet101', input_shape=input_shape, input_tensor=input_tensor,
                 include_top=include_top, num_classes=num_classes, weights=weights, **kwargs)


def SEResNet152(input_shape=None, input_tensor=None, weights=None, num_classes=1000, include_top=True, **kwargs):
    return SENet(MODELS_PARAMS['SEResNet152'], model_name='seresnet152', input_shape=input_shape, input_tensor=input_tensor,
                 include_top=include_top, num_classes=num_classes, weights=weights, **kwargs)


def SEResNeXt50(input_shape=None, input_tensor=None, weights=None, num_classes=1000, include_top=True, **kwargs):
    return SENet(MODELS_PARAMS['SEResNeXt50'], model_name='seresnext50', input_shape=input_shape, input_tensor=input_tensor,
                 include_top=include_top, num_classes=num_classes, weights=weights, **kwargs)


def SEResNeXt101(input_shape=None, input_tensor=None, weights=None, num_classes=1000, include_top=True, **kwargs):
    return SENet(MODELS_PARAMS['SEResNeXt101'], model_name='seresnext101', input_shape=input_shape, input_tensor=input_tensor,
                 include_top=include_top, num_classes=num_classes, weights=weights, **kwargs)


def SENet154(input_shape=None, input_tensor=None, weights=None, num_classes=1000, include_top=True, **kwargs):
    return SENet(MODELS_PARAMS['SENet154'], model_name='senet154', input_shape=input_shape, input_tensor=input_tensor,
                 include_top=include_top, num_classes=num_classes, weights=weights, **kwargs)


# ----------------------------------------------------------------------------------------------------------
# Set the attributes 
# ----------------------------------------------------------------------------------------------------------

setattr(SEResNet50, '__doc__', SENet.__doc__)
setattr(SEResNet101, '__doc__', SENet.__doc__)
setattr(SEResNet152, '__doc__', SENet.__doc__)
setattr(SEResNeXt50, '__doc__', SENet.__doc__)
setattr(SEResNeXt101, '__doc__', SENet.__doc__)
setattr(SENet154, '__doc__', SENet.__doc__)