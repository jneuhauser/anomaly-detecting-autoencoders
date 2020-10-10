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
from models.networks import Encoder, Decoder
from utils.model import print_model, print_layer, reset_weights


class CAE(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, **kwargs):
        kwargs['name'] = type(self).__name__
        super().__init__(**kwargs)

        # Use the DCGAN encoder/decoder models
        self.net_enc = Encoder(input_shape, latent_size, n_filters, n_extra_layers, name='encoder').model
        self.net_dec = Decoder(input_shape, latent_size, n_filters, n_extra_layers, name='decoder').model

    def call(self, x, training=False):
        encoded_image = self.net_enc(x, training=training)
        decoded_image = self.net_dec(encoded_image, training=training)
        return decoded_image

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
