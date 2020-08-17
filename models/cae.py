import numpy as np
import tensorflow as tf

import models.ae

print("TensorFlow Version: {}".format(tf.version.VERSION))

class CAE(models.ae.AE):
    def __init__(self, input_shape, conv_filters_list=(16,8)):
        encoder = self.create_encoder(input_shape, conv_filters_list)
        decoder = self.create_decoder(encoder)
        super().__init__(
            encoder=encoder,
            decoder=decoder
        )
    
    @staticmethod
    def create_encoder(input_shape, conv_filters_list):
        encoder = tf.keras.Sequential(name='encoder')
        encoder.add(
            tf.keras.layers.InputLayer(
                input_shape=input_shape
            )
        )
        for i, conv_filters in enumerate(conv_filters_list):
            encoder.add(
                tf.keras.layers.Conv2D(
                    name='conv-{}'.format(i),
                    filters=conv_filters,
                    kernel_size=(3,3),
                    strides=(1,1),
                    padding='same',
                    activation='relu'
                )
            )
            encoder.add(
                tf.keras.layers.MaxPool2D(
                    name='pool-{}'.format(i),
                    pool_size=(2,2),
                    strides=(2,2),
                    padding='same'
                )
            )
        return encoder
    
    @staticmethod
    def create_decoder(encoder):
        decoder = tf.keras.Sequential(name='decoder')
        decoder.add(
            tf.keras.layers.InputLayer(
                input_shape=encoder.layers[-1].output_shape[1:]
            )
        )
        is_conv_layer = lambda l: (
            issubclass(type(l), tf.keras.layers.Conv1D) or
            issubclass(type(l), tf.keras.layers.Conv2D) or
            issubclass(type(l), tf.keras.layers.Conv3D)
        )
        encoder_conv_layers = list(filter(lambda l: is_conv_layer(l), encoder.layers))
        for i, encoder_conv_layer in enumerate(reversed(encoder_conv_layers)):
            decoder.add(
                tf.keras.layers.Conv2DTranspose(
                    name='deconv-{}'.format(i),
                    filters=encoder_conv_layer.input_shape[-1],
                    kernel_size=(2,2),
                    strides=(2,2),
                    padding='same',
                    activation='relu'
                )
            )
        return decoder
