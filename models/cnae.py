# standard modules
import os
from packaging import version

import logging
logger   = logging.getLogger(__name__)
debug    = logger.debug
info     = logger.info
warning  = logger.warning
error    = logger.error
critical = logger.critical

# external modules
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# own modules
from models.networks import Encoder, Decoder, kernel_initializer
from utils.model import print_model, print_layer, reset_weights


class CNAE(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, **kwargs):
        kwargs['name'] = type(self).__name__
        super().__init__(**kwargs)

        # Use the DCGAN encoder/decoder models
        encoder = Encoder(input_shape, latent_size, n_filters, n_extra_layers, name='encoder').model
        decoder = Decoder(input_shape, latent_size, n_filters, n_extra_layers, name='decoder').model

        # build the encoder as simple sequential
        self.net_enc = tf.keras.Sequential(
            [
                # drop the latent convolution layer (last layer) from the encoder
                *encoder.layers[:-1],
                # preprocess before variational sampling
                tf.keras.layers.Flatten(name='encoder_flatten'),
                tf.keras.layers.Dense(latent_size, activation='relu', name='encoder_fc')
            ],
            name='encoder'
        )
        decoder_input_size = self.net_enc.layers[-3].output_shape[1:]

        # build the decoder as simple sequential
        self.net_dec = tf.keras.Sequential(
            [
                # postprocess after variational sampling
                tf.keras.layers.Dense(np.prod(decoder_input_size), activation='relu', name='decoder_fc'),
                tf.keras.layers.Reshape(decoder_input_size, name='decoder_reshape'),
                # drop the latent convolution layer with normalization and activation
                # (first three layers) from the decoder
                *decoder.layers[3:]
            ],
            name='decoder'
        )

        # Use input size as reconstruction loss weight
        self.loss_weight = tf.cast(tf.math.reduce_prod(input_shape), tf.float32)

    def summary(self, **kwargs):
        print_model(self)
        super().summary(**kwargs)
        self.net_enc.summary(**kwargs)
        self.net_dec.summary(**kwargs)

    def load_weights(self, path):
        if not (os.path.isfile(os.path.join(path, 'encoder.index')) and
                os.path.isfile(os.path.join(path, 'decoder.index'))):
            warning('No valid pre-trained network weights in: "{}"'.format(path))
            return
        self.net_enc.load_weights(os.path.join(path, 'encoder'))
        self.net_dec.load_weights(os.path.join(path, 'decoder'))
        info('Loaded pre-trained network weights from: "{}"'.format(path))

    def save_weights(self, path):
        self.net_enc.save_weights(os.path.join(path, 'encoder'))
        self.net_dec.save_weights(os.path.join(path, 'decoder'))
        info('Saved pre-trained network weights to: "{}"'.format(path))

    def call(self, x, training=False):
        x = self.net_enc(x, training=training)
        x = self.net_dec(x, training=training)
        return x
