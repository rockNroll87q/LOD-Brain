"""
Created on Feb 12 2021
@author: met
"""
from re import X
import tensorflow as tf
import tensorflow_addons as tfa


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num: int, dropout_rate: float, stride: int = 2, activation: str = 'relu', bn: bool = True,
                 groups=8, kernel_initializer='he_normal', kernel_regularizer=1.e-4, n_conv_row=1, **kwargs):
        """
        ResNet-like encoder bottleneck block for 3D tensors.

        Structure: - Input -|> Conv > BN > Conv > BN > Conv > BN > Dropout > Add -
                            \___________________> Conv > BN >_________________|
        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied downsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        """
        super(BottleNeck, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)

        self.conv1 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv2 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(3, 3, 3),
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv3 = tf.keras.layers.Conv3D(filters=filter_num * 4,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )

        if self.bn == 'BN':
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
        elif self.bn == 'GN':
            self.bn1 = tfa.layers.GroupNormalization(groups=groups)
            self.bn2 = tfa.layers.GroupNormalization(groups=groups)
            self.bn3 = tfa.layers.GroupNormalization(groups=groups)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv3D(filters=filter_num * 4,
                                                   kernel_size=(1, 1, 1),
                                                   strides=stride,
                                                   kernel_initializer=kernel_initializer,
                                                   kernel_regularizer=kernel_regularizer
                                                   )
                            )
        if self.bn:
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample.add(tfa.layers.GroupNormalization(groups=groups))

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        if self.bn in ['BN', 'GN']:
            x = self.bn1(x, training=training)
        x = getattr(tf.nn, self.activation)(x)
        x = self.conv2(x)
        if self.bn in ['BN', 'GN']:
            x = self.bn2(x, training=training)
        x = getattr(tf.nn, self.activation)(x)
        x = self.conv3(x)
        if self.bn in ['BN', 'GN']:
            x = self.bn3(x, training=training)
        if training:
            x = self.dropout(x)
        output = getattr(tf.nn, self.activation)(tf.keras.layers.add([residual, x]))

        return output

    def get_config(self):
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class Plain(tf.keras.layers.Layer):
    def __init__(self, filter_num: int, dropout_rate: float, stride: int = 2, activation: str = 'relu', bn: bool = True,
                 groups=8, kernel_initializer='he_normal', kernel_regularizer=1.e-4, n_conv_row=1, **kwargs):
        """
        VGG-like encoder block for 3D tensors.

        Structure: - Input -|> Conv > Conv (DS) > BN > Dropout -

        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied downsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        """
        super(Plain, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
        self.n_conv_row = n_conv_row

        # Define conv layers with Convs, BN, and activation
        self.convs = tf.keras.Sequential()
        for i in range(self.n_conv_row):
            self.convs.add(tf.keras.layers.Conv3D(filters=filter_num,       # Conv
                                            kernel_size=(3, 3, 3),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            ))
            if self.bn == 'BN':                                             # Batch norm
                self.convs.add(tf.keras.layers.BatchNormalization())
            elif self.bn == 'GN':
                self.convs.add(tfa.layers.GroupNormalization(groups=groups))
            self.convs.add(tf.keras.layers.Activation(self.activation))     # Activation

        self.down_conv = tf.keras.layers.Conv3D(filters=4 * filter_num,
                                                kernel_size=(stride, stride, stride),
                                                strides=stride,
                                                padding='same',
                                                kernel_initializer=kernel_initializer,
                                                kernel_regularizer=kernel_regularizer
                                                )

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.convs(inputs, training=training)
        if training:
            x = self.dropout(x)                        # dropout
        x = self.down_conv(x)
        x = getattr(tf.nn, self.activation)(x)        
        return x

    def get_config(self):
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "n_conv_row": self.n_conv_row,
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class UpBottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num: int, dropout_rate: float, stride: int = 2, activation: str = 'relu', bn: bool = True,
                 groups=8, kernel_initializer='he_normal', kernel_regularizer=1.e-4, n_conv_row=1, **kwargs):
        """
        ResNet-like decoder bottleneck block for 3D tensors.

        Structure: - Input -|> Conv > BN > ConvT > BN > Conv > BN > Dropout > Add -
                            \___________________> ConvT > BN >_________________|
        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied upsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        """
        super(UpBottleNeck, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)

        self.conv1 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv2_up = tf.keras.layers.Conv3DTranspose(filters=filter_num,
                                                        kernel_size=(3, 3, 3),
                                                        strides=stride,
                                                        padding='same',
                                                        kernel_initializer=kernel_initializer,
                                                        kernel_regularizer=kernel_regularizer
                                                        )
        self.conv3 = tf.keras.layers.Conv3D(filters=filter_num * 4,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer)

        if self.bn == 'BN':
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
        elif self.bn == 'GN':
            self.bn1 = tfa.layers.GroupNormalization(groups=groups)
            self.bn2 = tfa.layers.GroupNormalization(groups=groups)
            self.bn3 = tfa.layers.GroupNormalization(groups=groups)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.upsample = tf.keras.Sequential()
        self.upsample.add(tf.keras.layers.Conv3DTranspose(filters=filter_num * 4,
                                                          kernel_size=(1, 1, 1),
                                                          strides=stride,
                                                          kernel_initializer=kernel_initializer,
                                                          kernel_regularizer=kernel_regularizer
                                                          )
                          )
        if self.bn:
            self.upsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.upsample.add(tfa.layers.GroupNormalization(groups=groups))

    def call(self, inputs, training=None, **kwargs):
        residual = self.upsample(inputs)

        x = self.conv1(inputs)
        if self.bn in ['BN', 'GN']:
            x = self.bn1(x, training=training)
        x = getattr(tf.nn, self.activation)(x)
        x = self.conv2_up(x)
        if self.bn in ['BN', 'GN']:
            x = self.bn2(x, training=training)
        x = getattr(tf.nn, self.activation)(x)
        x = self.conv3(x)
        if self.bn in ['BN', 'GN']:
            x = self.bn3(x, training=training)
        if training:
            x = self.dropout(x)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

    def get_config(self):
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


class UpPlain(tf.keras.layers.Layer):
    def __init__(self, filter_num: int, dropout_rate: float, stride: int = 2, activation: str = 'relu', bn: bool = True,
                 groups=8, kernel_initializer='he_normal', kernel_regularizer=1.e-4, n_conv_row=1,  **kwargs):
        """
        VGG-like encoder block for 3D tensors.

        Structure: - Input -|> Conv > Conv Transpose > BN > Dropout -

        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied downsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        """
        super(UpPlain, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
        self.n_conv_row = n_conv_row

        # Define conv layers with Convs, BN, and activation
        self.convs = tf.keras.Sequential()
        for i in range(self.n_conv_row):
            self.convs.add(tf.keras.layers.Conv3D(filters=filter_num,       # Conv
                                            kernel_size=(3, 3, 3),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            ))
            if self.bn == 'BN':                                             # Batch norm
                self.convs.add(tf.keras.layers.BatchNormalization())
            elif self.bn == 'GN':
                self.convs.add(tfa.layers.GroupNormalization(groups=groups))
            self.convs.add(tf.keras.layers.Activation(self.activation))     # Activation

        self.up_conv = tf.keras.layers.Conv3DTranspose(filters=filter_num * 4,
                                                       kernel_size=(stride, stride, stride),
                                                       strides=stride,
                                                       padding='same',
                                                       kernel_initializer=kernel_initializer,
                                                       kernel_regularizer=kernel_regularizer
                                                       )

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.convs(inputs, training=training)
        if training:
            x = self.dropout(x)
        x = self.up_conv(x)
        x = getattr(tf.nn, self.activation)(x)
        return x

    def get_config(self):
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "n_conv_row": self.n_conv_row,
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


    #     # Define layers
    #     self.convs = []
    #     self.bns = []
    #     for i in range(self.n_conv_row):
    #         self.convs.append(tf.keras.layers.Conv3D(filters=filter_num,
    #                                         kernel_size=(3, 3, 3),
    #                                         strides=1,
    #                                         padding='same',
    #                                         kernel_initializer=kernel_initializer,
    #                                         kernel_regularizer=kernel_regularizer,
    #                                         ))

    #         if self.bn == 'BN':
    #             self.bns.append(tf.keras.layers.BatchNormalization())
    #         elif self.bn == 'GN':
    #             self.bns.append(tfa.layers.GroupNormalization(groups=groups))

    #     self.down_conv = tf.keras.layers.Conv3D(filters=4 * filter_num,
    #                                             kernel_size=(stride, stride, stride),
    #                                             strides=stride,
    #                                             padding='same',
    #                                             kernel_initializer=kernel_initializer,
    #                                             kernel_regularizer=kernel_regularizer
    #                                             )

    #     self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    # def call(self, inputs, training=None, **kwargs):

    #     x = inputs
    #     for i_conv, i_bn in zip(self.convs, self.bns):
    #         x = i_conv(x)                               # conv
    #         if self.bn in ['BN', 'GN']:                 # bn
    #             x = i_bn(x, training=training)
    #         x = getattr(tf.nn, self.activation)(x)      # relu
        
    #     if training:
    #         x = self.dropout(x)                        # dropout

    #     x = self.down_conv(x)
    #     x = getattr(tf.nn, self.activation)(x)        

    #     return x

    # def get_config(self):
    #     return {"filter_num": self.filter_num,
    #             "dropout_rate": self.dropout_rate,
    #             "stride": self.stride,
    #             "activation": self.activation,
    #             "bn": self.bn,
    #             "n_conv_row": self.n_conv_row,
    #             }