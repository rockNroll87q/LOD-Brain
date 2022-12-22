"""
Created on Feb 12 2021
@authors:
* Mattia Savardi, University of Brescia
* Michele Svanera, University of Glasgow
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from loguru import logger
import numpy as np
from LOD_Brain.model.layers import BottleNeck, UpBottleNeck, Plain, UpPlain
from LOD_Brain.config import NetConfig as Config


def build_flat_model(config: Config, ds_factor: int = 1) -> Model:
    """
    Build the 3D segmentation model.

    :param config: pydantic config object with the parameters.
    :param ds_factor: apply a downsampling of factor `ds_factor`
    :return: keras model.
    """
    assert np.log2(config.size // ds_factor) >= len(config.num_filters), \
        "The given number of filters and levels is more than what the current input size supports"

    num_filters = config.num_filters
    # with tf.python.keras.backend.get_graph().as_default():  # bugfix for tf 2.2.0. see also https://github.com/tensorflow/tensorflow/issues/27298
    inputs = tf.keras.layers.Input((config.size, config.size, config.size, 1))

    if ds_factor != 1:
        logger.debug(f"Applying a pre downsampling of factor {ds_factor}")
        ds_inputs = tf.keras.layers.MaxPooling3D(pool_size=(ds_factor, ds_factor, ds_factor))(inputs)
        inputs_resized = ds_inputs
    else:
        inputs_resized = inputs

    conv_blocks = {"BottleNeck": {"Encoder": BottleNeck, "Decoder": UpBottleNeck},
                   "Plain": {"Encoder": Plain, "Decoder": UpPlain}
                   }
    conv_block = conv_blocks[config.conv_block]

    skip_x = []
    skip_up = []

    x = inputs_resized

    # Encoder
    with tf.name_scope(f"Encoder"):
        x = tf.keras.layers.Conv3D(filters=4 * num_filters[0],
                                   kernel_size=(3, 3, 3),
                                   strides=(1, 1, 1),
                                   padding='same',
                                   name=f"enc_{4 * num_filters[0]}")(x)
        skip_x.append(x)
        for i, f in enumerate(num_filters[1:-1]):
            x = conv_block["Encoder"](filter_num=f, dropout_rate=config.dropout_rate, stride=config.stride,
                                      activation=config.activation_enc, bn=config.bn, groups=min(8, f),
                                      name=f"enc_cb_{f}")(x)
            skip_x.append(x)

        # Bridge
        x = conv_block["Encoder"](filter_num=num_filters[-1], dropout_rate=config.dropout_rate,
                                  stride=config.stride, activation=config.activation_enc, bn=config.bn,
                                  groups=min(8, num_filters[-1]),
                                  name=f"enc_cb_{num_filters[-1]}")(x)

    num_filters.reverse()
    skip_x.reverse()

    # Decoder
    with tf.name_scope(f"Decoder"):
        for i, f in enumerate(num_filters[1:]):
            x = conv_block["Decoder"](filter_num=f, dropout_rate=config.dropout_rate, stride=config.stride,
                                      activation=config.activation_dec, bn=config.bn, groups=min(8, f),
                                      name=f"dec_cb_{f}")(x)
            x = tf.keras.layers.Add()([x, skip_x[i]])
            skip_up.append(x)

    # Output
    with tf.name_scope(f"Output"):
        x = tf.keras.layers.Conv3D(filters=4 * num_filters[-1],  # config.num_classes,
                                   kernel_size=(3, 3, 3),
                                   padding='same',
                                   kernel_initializer=config.kernel_initializer,
                                   kernel_regularizer=tf.keras.regularizers.L2(config.kernel_regularizer),
                                   name=f"out_cb_{4 * num_filters[-1]}")(x)

        num_filters.reverse()

    # Restore initial output size and apply the final activation (SoftMax)
    if ds_factor != 1:
        x = tf.keras.layers.UpSampling3D(size=ds_factor)(x)
    x = tf.keras.layers.Conv3D(filters=config.num_classes,
                               kernel_size=(1, 1, 1),
                               activation='softmax',
                               kernel_initializer=config.kernel_initializer,
                               kernel_regularizer=tf.keras.regularizers.L2(config.kernel_regularizer),
                               name=f'main_output_ds{ds_factor}')(x)

    return Model(inputs, x)


def build_vanilla_unet_model(config: Config) -> Model:
    """
    Build the 3D segmentation model.

    :param config: pydantic config object with the parameters.
    :return: keras model.
    """

    num_filters = config.num_filters
    # with tf.python.keras.backend.get_graph().as_default():  # bugfix for tf 2.2.0. see also https://github.com/tensorflow/tensorflow/issues/27298
    inputs = tf.keras.layers.Input((config.size, config.size, config.size, 1))

    inputs_resized = inputs
    conv_blocks = {"BottleNeck": {"Encoder": BottleNeck, "Decoder": UpBottleNeck},
                   "Plain": {"Encoder": Plain, "Decoder": UpPlain}
                   }
    conv_block = conv_blocks[config.conv_block]

    skip_x = []

    x = inputs_resized

    # Encoder
    with tf.name_scope(f"Encoder"):
        x = tf.keras.layers.Conv3D(filters=4 * num_filters[0],
                                   kernel_size=(3, 3, 3),
                                   strides=(1, 1, 1),
                                   padding='same',
                                   name=f"enc_{4 * num_filters[0]}")(x)
        skip_x.append(x)
        for i, f in enumerate(num_filters[1:-1]):
            x = conv_block["Encoder"](filter_num=f, dropout_rate=config.dropout_rate, stride=config.stride,
                                      activation=config.activation_enc, bn=config.bn, groups=min(8, f),
                                      name=f"enc_cb_{f}")(x)
            skip_x.append(x)

        # Bridge
        x = conv_block["Encoder"](filter_num=num_filters[-1], dropout_rate=config.dropout_rate,
                                  stride=config.stride, activation=config.activation_enc, bn=config.bn,
                                  groups=min(8, num_filters[-1]),
                                  name=f"enc_cb_{num_filters[-1]}")(x)

    num_filters.reverse()
    skip_x.reverse()

    # Decoder
    with tf.name_scope(f"Decoder"):
        for i, f in enumerate(num_filters[1:]):
            x = conv_block["Decoder"](filter_num=f, dropout_rate=config.dropout_rate, stride=config.stride,
                                      activation=config.activation_dec, bn=config.bn, groups=min(8, f),
                                      name=f"dec_cb_{f}")(x)
            x = tf.keras.layers.Add()([x, skip_x[i]])

    # Output
    with tf.name_scope(f"Output"):
        x = tf.keras.layers.Conv3D(filters=4 * num_filters[-1],  # config.num_classes,
                                   kernel_size=(3, 3, 3),
                                   padding='same',
                                   kernel_initializer=config.kernel_initializer,
                                   kernel_regularizer=tf.keras.regularizers.L2(config.kernel_regularizer),
                                   name=f"out_cb_{4 * num_filters[-1]}")(x)

        num_filters.reverse()

    # Restore initial output size and apply the final activation (SoftMax)
    x = tf.keras.layers.Conv3D(filters=config.num_classes,
                               kernel_size=(1, 1, 1),
                               activation='softmax',
                               kernel_initializer=config.kernel_initializer,
                               kernel_regularizer=tf.keras.regularizers.L2(config.kernel_regularizer),
                               name=f'main_output')(x)

    return Model(inputs, x)


def build_cerebrum_model(config: Config) -> Model:
    """
    Build the 3D segmentation model.

    :param config: pydantic config object with the parameters.
    :return: keras model.
    """

    n_filters = 48
    max_n_filters = 48 * 4
    lvl1_dropout = 0.0
    lvl2_dropout = 0.25
    lvl3_dropout = 0.5

    lvl1_kernel_size = (3, 3, 3)
    lvl2_kernel_size = (3, 3, 3)
    lvl3_kernel_size = (3, 3, 3)
    
    encoder_act_function = 'relu'
    decoder_act_function = 'relu'
    init = 'glorot_normal'
    final_activation = 'softmax'
    n_classes = config.num_classes

    inputs = tf.keras.layers.Input((config.size, config.size, config.size, 1))

    # -----------------------------------
    #         ENCODER - LEVEL 1
    # -----------------------------------
    
    # Level1-block1 conv layer: 1-conv (full-resolution) and dropout
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    enc_lvl1_block1_conv = tf.keras.layers.Conv3D(filters = n_filters, 
                                    kernel_size = lvl1_kernel_size, 
                                    strides = (1,1,1), 
                                    padding = 'same', 
                                    activation = encoder_act_function, 
                                    kernel_initializer = init, 
                                    name = 'enc_lvl1_block1_conv')(inputs)
    enc_lvl1_block1_dr = tf.keras.layers.Dropout(rate = lvl1_dropout, 
                                    name = 'enc_lvl1_block1_dr')(enc_lvl1_block1_conv)

    # Level1 strided-conv layer: down-sampling
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    down_1to2_conv = tf.keras.layers.Conv3D(filters = min(2*n_filters,max_n_filters), 
                            kernel_size = (4,4,4), 
                            strides = (4,4,4), 
                            padding = 'same', 
                            activation = encoder_act_function, 
                            kernel_initializer = init, 
                            name = 'down_1to2_conv')(enc_lvl1_block1_dr)
    down_1to2_dr = tf.keras.layers.Dropout(rate = lvl1_dropout, 
                            name = 'down_1to2_dr')(down_1to2_conv)

    # -----------------------------------
    #         ENCODER - LEVEL 2
    # -----------------------------------

    # Level2-block1 conv layer: 2-conv (1/4 resolution) and dropout
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    enc_lvl2_block1_conv = tf.keras.layers.Conv3D(filters = min(2*n_filters,max_n_filters), 
                                    kernel_size = lvl2_kernel_size, 
                                    strides = (1,1,1), 
                                    padding = 'same', 
                                    activation = encoder_act_function, 
                                    kernel_initializer = init, 
                                    name = 'enc_lvl2_block1_conv')(down_1to2_dr)
    enc_lvl2_block1_dr = tf.keras.layers.Dropout(rate = lvl2_dropout, 
                                    name = 'enc_lvl2_block1_dr')(enc_lvl2_block1_conv)
    
    # Level2-block2 conv layer: 2-conv (1/4 resolution) and dropout
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    enc_lvl2_block2_conv = tf.keras.layers.Conv3D(filters = min(2*n_filters,max_n_filters), 
                                    kernel_size = lvl2_kernel_size, 
                                    strides = (1,1,1), 
                                    padding = 'same', 
                                    activation = encoder_act_function, 
                                    kernel_initializer = init, 
                                    name = 'enc_lvl2_block2_conv')(enc_lvl2_block1_dr)
    enc_lvl2_block2_dr = tf.keras.layers.Dropout(rate = lvl2_dropout, 
                                    name = 'enc_lvl2_block2_dr')(enc_lvl2_block2_conv)

    # Level2 strided-conv layer: down-sampling
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    down_2to3_conv = tf.keras.layers.Conv3D(filters = min(4*n_filters,max_n_filters), 
                            kernel_size = (2,2,2), 
                            strides = (2,2,2), 
                            padding = 'same', 
                            activation = encoder_act_function, 
                            kernel_initializer = init, 
                            name = 'down_2to3_conv')(enc_lvl2_block2_dr)
    down_2to3_dr = tf.keras.layers.Dropout(rate = lvl2_dropout, 
                            name = 'down_2to3_dr')(down_2to3_conv)

                                
    # -----------------------------------
    #         BOTTLENECK LAYER (Layer3)
    # -----------------------------------
                                            
    # Level3-block1 conv layer: 3-conv (1/8 resolution) and dropout
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    enc_lvl3_block1_conv = tf.keras.layers.Conv3D(filters = min(4*n_filters,max_n_filters), 
                                    kernel_size = lvl3_kernel_size, 
                                    strides = (1,1,1), 
                                    padding = 'same', 
                                    activation = encoder_act_function, 
                                    kernel_initializer = init, 
                                    name = 'enc_lvl3_block1_conv')(down_2to3_dr)
    enc_lvl3_block1_dr = tf.keras.layers.Dropout(rate = lvl3_dropout, 
                                    name = 'enc_lvl3_block1_dr')(enc_lvl3_block1_conv)

    # Level3-block2 conv layer: 3-conv (1/8 resolution) and dropout
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    enc_lvl3_block2_conv = tf.keras.layers.Conv3D(filters = min(4*n_filters,max_n_filters), 
                                    kernel_size = lvl3_kernel_size, 
                                    strides = (1,1,1), 
                                    padding = 'same', 
                                    activation = encoder_act_function, 
                                    kernel_initializer = init, 
                                    name = 'enc_lvl3_block2_conv')(enc_lvl3_block1_dr)
    enc_lvl3_block2_dr = tf.keras.layers.Dropout(rate = lvl3_dropout, 
                                    name = 'enc_lvl3_block2_dr')(enc_lvl3_block2_conv)
    
    # Level3-block3 conv layer: 3-conv (1/8 resolution) and dropout
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    enc_lvl3_block3_conv = tf.keras.layers.Conv3D(filters = min(4*n_filters,max_n_filters), 
                                    kernel_size = lvl3_kernel_size, 
                                    strides = (1,1,1), 
                                    padding = 'same', 
                                    activation = decoder_act_function, 
                                    kernel_initializer = init, 
                                    name = 'enc_lvl3_block3_conv')(enc_lvl3_block2_dr)
    enc_lvl3_block3_dr = tf.keras.layers.Dropout(rate = lvl3_dropout, 
                                    name = 'enc_lvl3_block3_dr')(enc_lvl3_block3_conv)


    # Level3 strided-conv layer: up-sampling
    # ... -> Transpose Conv 3D -> Activation (None) -> BatchNorm (NO) -> Dropout -> ...
    up_3to2_conv = tf.keras.layers.Conv3DTranspose(filters = min(2*n_filters,max_n_filters), 
                                    kernel_size = (2,2,2), 
                                    strides = (2,2,2), 
                                    padding = 'same', 
                                    activation = None, 
                                    kernel_initializer = init, 
                                    name = 'up_3to2_conv')(enc_lvl3_block3_dr)
    up_3to2_dr = tf.keras.layers.Dropout(rate = lvl2_dropout, 
                            name = 'up_3to2_dr')(up_3to2_conv)

    
    # -----------------------------------
    #         DECODER - LEVEL 2
    # -----------------------------------

    # Level2 Skip-connection: last of second enc. level + last of third enc. level
    skip_conn_lvl2 = tf.keras.layers.Add(name = 'skip_conn_lvl2')([enc_lvl2_block2_dr, up_3to2_dr])


    # Level2-block1 conv layer: 2-conv (1/4 resolution) and dropout
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    dec_lvl2_block1_conv = tf.keras.layers.Conv3D(filters = min(2*n_filters,max_n_filters), 
                                    kernel_size = lvl2_kernel_size, 
                                    strides = (1,1,1), 
                                    padding = 'same', 
                                    activation = decoder_act_function, 
                                    kernel_initializer = init, 
                                    name = 'dec_lvl2_block1_conv')(skip_conn_lvl2)
    dec_lvl2_block1_dr = tf.keras.layers.Dropout(rate = lvl2_dropout, 
                                    name = 'dec_lvl2_block1_dr')(dec_lvl2_block1_conv)
    
    # Level2-block2 conv layer: 2-conv (1/4 resolution) and dropout
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    dec_lvl2_block2_conv = tf.keras.layers.Conv3D(filters = min(2*n_filters,max_n_filters), 
                                    kernel_size = lvl2_kernel_size, 
                                    strides = (1,1,1), 
                                    padding = 'same', 
                                    activation = decoder_act_function, 
                                    kernel_initializer = init, 
                                    name = 'dec_lvl2_block2_conv')(dec_lvl2_block1_dr)
    dec_lvl2_block2_dr = tf.keras.layers.Dropout(rate = lvl2_dropout, 
                                    name = 'dec_lvl2_block2_dr')(dec_lvl2_block2_conv)


    # Level2 strided-conv layer: up-sampling
    # ... -> Transpose Conv 3D -> Activation (None) -> BatchNorm (NO) -> Dropout -> ...
    up_2to1_conv = tf.keras.layers.Conv3DTranspose(filters = n_filters, 
                                    kernel_size = (4,4,4), 
                                    strides = (4,4,4), 
                                    padding = 'same', 
                                    activation = None, 
                                    kernel_initializer = init, 
                                    name = 'up_2to1_conv')(dec_lvl2_block2_dr)
    up_2to1_dr = tf.keras.layers.Dropout(rate = lvl2_dropout, 
                            name = 'up_2to1_dr')(up_2to1_conv)

    # -----------------------------------
    #         DECODER - LEVEL 1
    # -----------------------------------

    # Level1 Skip-connection: last of first enc. level + last of second dec. level
    skip_conn_lvl1 = tf.keras.layers.Add(name = 'skip_conn_lvl1')([enc_lvl1_block1_dr, up_2to1_dr])


    # Level1-block1 conv layer: 1-conv (full-resolution) and dropout
    # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
    dec_lvl1_block1_conv = tf.keras.layers.Conv3D(filters = n_filters, 
                                    kernel_size = lvl1_kernel_size, 
                                    strides = (1,1,1), 
                                    padding = 'same', 
                                    activation = decoder_act_function, 
                                    kernel_initializer = init, 
                                    name = 'dec_lvl1_block1_conv')(skip_conn_lvl1)
    dec_lvl1_block1_dr = tf.keras.layers.Dropout(rate = lvl1_dropout, 
                                    name = 'dec_lvl1_block1_dr')(dec_lvl1_block1_conv)

    # Output data
    outputs = tf.keras.layers.Conv3D(filters = n_classes, 
                        kernel_size = (1,1,1), 
                        activation = final_activation,
                        kernel_initializer = init, 
                        name = 'main_output')(dec_lvl1_block1_dr)

    # Define the model object, the optimizer, and compile them
    return Model(inputs, outputs)   


def build_model(config: Config, ds_factor: int = 1) -> Model:
    """
    Build the 3D segmentation model.

    :param config: pydantic config object with the parameters (config.network).
    :param ds_factor: apply a downsampling of factor `ds_factor`
    :return: keras model.
    """
    assert np.log2(config.size // ds_factor / (2 ** (config.num_levels - 1))) >= len(config.num_filters), \
        "The given number of filters and levels is more than what the current input size supports"
    num_levels = min(len(config.num_filters) - 1, config.num_levels)
    if num_levels != config.num_levels:
        logger.warning(
            f"The maximum allowed number of levels, given the provided filters {config.num_filters} is {num_levels}. "
            f"Setting the number of level to {config.num_filters}.")
    logger.debug(f"Levels: {config.num_levels} >>> {num_levels}, filters: {config.num_filters}")
    num_filters = config.num_filters
    # with tf.python.keras.backend.get_graph().as_default():  # bugfix for tf 2.2.0. see also https://github.com/tensorflow/tensorflow/issues/27298
    inputs = tf.keras.layers.Input((config.size, config.size, config.size, 1))
    inputs_resized = [tf.keras.layers.MaxPooling3D(pool_size=(ds_factor * (2 ** factor),
                                                              ds_factor * (2 ** factor),
                                                              ds_factor * (2 ** factor)))(inputs) for
                      factor in range(num_levels - 1, 0, -1)]
    if ds_factor != 1:
        logger.debug(f"Applying a pre downsampling of factor {ds_factor}")
        ds_inputs = tf.keras.layers.MaxPooling3D(pool_size=(ds_factor, ds_factor, ds_factor))(inputs)
        inputs_resized.append(ds_inputs)
    else:
        inputs_resized.append(inputs)

    conv_blocks = {"BottleNeck": {"Encoder": BottleNeck, "Decoder": UpBottleNeck},
                   "Plain": {"Encoder": Plain, "Decoder": UpPlain}
                   }
    conv_block = conv_blocks[config.conv_block]

    skip_x = []
    skip_up = []
    outputs = []
    conv_block_repetition = 1
    for lv in range(num_levels):
        if config.filter_multiplicative_factors and len(config.filter_multiplicative_factors)>=lv:
            filter_multiplicative_factor = config.filter_multiplicative_factors[lv]
        else:
            filter_multiplicative_factor = 1
        skip_x_lv = []
        x = inputs_resized[lv]
        logger.debug(f"--- Level {lv} ---")
        with tf.name_scope(f"Level_{lv}"):
            # Encoder
            with tf.name_scope(f"Level_{lv}_encoder"):
                if config.boost_first_layer and filter_multiplicative_factor !=1:
                    x = tf.keras.layers.Conv3D(filters=4 * num_filters[0] * filter_multiplicative_factor,
                                            kernel_size=(3, 3, 3),
                                            strides=(1, 1, 1),
                                            # activation=config.activation_enc,
                                            padding='same')(x)
                    x = tf.keras.layers.Conv3D(filters=4 * num_filters[0],
                                                    kernel_size=(1, 1, 1),
                                                    strides=(1, 1, 1),
                                                    padding='same')(x)
                else:
                    x = tf.keras.layers.Conv3D(filters=4 * num_filters[0],
                                            kernel_size=(3, 3, 3),
                                            strides=(1, 1, 1),
                                            padding='same')(x)

                skip_x_lv.append(x)
                for i, f in enumerate(num_filters[1:-1 - lv]):
                    x = conv_block["Encoder"](filter_num=f, dropout_rate=config.dropout_rate, stride=config.stride,
                                              activation=config.activation_enc, bn=config.bn, groups=min(8, f), 
                                              n_conv_row=min(conv_block_repetition, 2),
                                              mult_factor=filter_multiplicative_factor)(x)
                    if config.conv_repetition:
                        conv_block_repetition += 1                                              
                    if lv:
                        if config.upskip == 'Conv3DTranspose':
                            ups = tf.keras.layers.Conv3DTranspose(filters=1,  # consider to use the #filters of x
                                                                  kernel_size=(3, 3, 3),
                                                                  strides=config.stride,
                                                                  padding='same',
                                                                  kernel_initializer=config.kernel_initializer,
                                                                  kernel_regularizer=tf.keras.regularizers.L2(
                                                                      config.kernel_regularizer)
                                                                  )(skip_x[lv - 1][i + 1])
                        else:
                            ups = tf.keras.layers.UpSampling3D(size=2)(skip_x[lv - 1][i + 1])
                        x = tf.keras.layers.Add()([x, ups])
                    skip_x_lv.append(x)
                skip_x.append(skip_x_lv)

                # Bridge
                x = conv_block["Encoder"](filter_num=num_filters[-1], dropout_rate=config.dropout_rate,
                                          stride=config.stride, activation=config.activation_enc, bn=config.bn,
                                          groups=min(8, num_filters[-1]), n_conv_row=min(conv_block_repetition,2))(x)
                if config.conv_repetition:
                    conv_block_repetition += 1

            num_filters.reverse()
            skip_x_lv.reverse()

            # Decoder
            with tf.name_scope(f"Level_{lv}_decoder"):
                skip_up_lv = []
                for i, f in enumerate(num_filters[1 + lv:]):
                    if config.conv_repetition:
                        conv_block_repetition -= 1
                    x = conv_block["Decoder"](filter_num=f, dropout_rate=config.dropout_rate, stride=config.stride,
                                              activation=config.activation_dec, bn=config.bn, groups=min(8, f), 
                                              n_conv_row=min(conv_block_repetition,2),
                                              mult_factor=filter_multiplicative_factor)(x)
                    x = tf.keras.layers.Add()([x, skip_x_lv[i]])
                    if lv:
                        if config.upskip == 'Conv3DTranspose':
                            ups = tf.keras.layers.Conv3DTranspose(filters=1,  # consider to use the #filters of x
                                                                  kernel_size=(3, 3, 3),
                                                                  strides=config.stride,
                                                                  padding='same',
                                                                  kernel_initializer=config.kernel_initializer,
                                                                  kernel_regularizer=tf.keras.regularizers.L2(
                                                                      config.kernel_regularizer)
                                                                  )(skip_up[lv - 1][i + 1])
                        else:
                            ups = tf.keras.layers.UpSampling3D(size=2)(skip_up[lv - 1][i + 1])
                        x = tf.keras.layers.Add()([x, ups])
                    skip_up_lv.append(x)
                skip_up.append(skip_up_lv)

            # Output
            with tf.name_scope(f"Level_{lv}_output"):
                x = tf.keras.layers.Conv3D(filters=4 * num_filters[-1],  # config.num_classes,
                                           kernel_size=(3, 3, 3),
                                           padding='same',
                                           kernel_initializer=config.kernel_initializer,
                                           kernel_regularizer=tf.keras.regularizers.L2(config.kernel_regularizer))(x)

                if (num_levels - lv - 1):           # if full resolution (256-iso), does not upsample
                    x = tf.keras.layers.UpSampling3D(size=2 ** (num_levels - lv - 1))(x)
                if not config.skip_intermediate_result:
                    outputs.append(x)
                num_filters.reverse()
                skip_x_lv.reverse()

    if num_levels > 1 and not config.skip_intermediate_result:
        x = tf.keras.layers.Add()(outputs)

    # Restore initial output size and apply the final activation (SoftMax)
    if ds_factor != 1:
        x = tf.keras.layers.UpSampling3D(size=ds_factor)(x)
    x = tf.keras.layers.Conv3D(filters=config.num_classes,
                               kernel_size=(1, 1, 1),
                               activation='softmax',
                               kernel_initializer=config.kernel_initializer,
                               kernel_regularizer=tf.keras.regularizers.L2(config.kernel_regularizer),
                               name='main_output')(x)

    return Model(inputs, x)
