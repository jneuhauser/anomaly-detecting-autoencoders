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
        encoded_image = self.net_enc(x, training=training)
        decoded_image = self.net_dec(encoded_image, training=training)
        return decoded_image

    def train_step(self, data):
        x, _, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        return super().train_step((x, x, sample_weight))

    def test_step(self, data):
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)

        x_pred = self(x, training=False)

        #loss = self.compiled_loss(
        #        x,
        #        decoded_image,
        #        sample_weight=sample_weight,
        #        regularization_losses=self.losses,
        #    )

        # we need a loss value per image which isn't provided by compiled_loss
        # assume we always have a shape of (batch_size, width, height, depth)
        losses = tf.keras.backend.mean(tf.keras.backend.square(x - x_pred), axis=[1,2,3])

        return {
            "losses": tf.reshape(losses, (-1, 1)),
            "labels": tf.reshape(y, (-1, 1))
            }
