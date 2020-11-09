# standard modules
import sys
assert sys.version_info >= (3, 5), "Python 3.5 or greater required"
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

# Based on: https://keras.io/examples/generative/vae/

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class CVAE(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, intermediate_size=0, **kwargs):
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
                tf.keras.layers.Flatten(name='encoder_flatten')
            ],
            name='encoder'
        )
        variational_input_size = (np.prod(self.net_enc.layers[-2].output_shape[1:]),)
        decoder_input_size = self.net_enc.layers[-2].output_shape[1:]
        # add an optional fully connected intermediate layer
        if intermediate_size and intermediate_size > 0:
            variational_input_size = (intermediate_size,)
            self.net_enc.add(tf.keras.layers.Dense(
                intermediate_size, activation='relu', name='encoder_intermediate'))

        # build the variational part with the functional api
        variational_input = tf.keras.Input(shape=variational_input_size, name='input_variational')
        z_mean = tf.keras.layers.Dense(latent_size, name='z_mean')(variational_input)
        z_log_var = tf.keras.layers.Dense(latent_size, name='z_log_var')(variational_input)
        # sample z from z_mean and z_log_var
        z = Sampling(name='sampling_z')([z_mean, z_log_var])
        self.net_var = tf.keras.Model(variational_input, [z_mean, z_log_var, z], name='variational')

        # build the decoder as simple sequential
        self.net_dec = tf.keras.Sequential(
            [
                # postprocess after variational sampling
                tf.keras.layers.Dense(np.prod(decoder_input_size), activation='relu', name='decoder_intermediate'),
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
        self.net_var.summary(**kwargs)
        self.net_dec.summary(**kwargs)

    def load_weights(self, path):
        if not (os.path.isfile(os.path.join(path, 'encoder.index')) and
                os.path.isfile(os.path.join(path, 'variational.index')) and
                os.path.isfile(os.path.join(path, 'decoder.index'))):
            warning('No valid pre-trained network weights in: "{}"'.format(path))
            return
        self.net_enc.load_weights(os.path.join(path, 'encoder'))
        self.net_var.load_weights(os.path.join(path, 'variational'))
        self.net_dec.load_weights(os.path.join(path, 'decoder'))
        info('Loaded pre-trained network weights from: "{}"'.format(path))

    def save_weights(self, path):
        self.net_enc.save_weights(os.path.join(path, 'encoder'))
        self.net_var.save_weights(os.path.join(path, 'variational'))
        self.net_dec.save_weights(os.path.join(path, 'decoder'))
        info('Saved pre-trained network weights to: "{}"'.format(path))


    def call(self, x, training=False):
        encoded_image = self.net_enc(x, training=training)
        _, _, z = self.net_var(encoded_image, training=training)
        decoded_image = self.net_dec(z, training=training)
        return decoded_image

    def train_step(self, data):
        x, _, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            encoded_image = self.net_enc(x, training=True)
            z_mean, z_log_var, z = self.net_var(encoded_image, training=True)
            decoded_image = self.net_dec(z, training=True)

            #reconstruction_loss = tf.reduce_mean(
            #    tf.keras.losses.binary_crossentropy(x, decoded_image)
            #    #tf.keras.losses.MSE(x, decoded_image)
            #    #tf.keras.losses.MAE(x, decoded_image)
            #)
            loss = self.compiled_loss(
                x,
                decoded_image,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5

            # Use input size as reconstruction loss weight
            loss_weight = tf.cast(tf.math.reduce_prod(tf.shape(x)[1:]), tf.float32)

            total_loss = loss * loss_weight + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.compiled_metrics.update_state(x, decoded_image, sample_weight=sample_weight)

        return {
            **{
                'total_loss': total_loss,
                #'rec_loss': reconstruction_loss, # is included in metrics dict below
                'kl_loss': kl_loss
            },
            **{m.name: m.result() for m in self.metrics}
        }

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

    def predict_step(self, data):
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)

        x_pred = self(x, training=False)

        losses = tf.keras.backend.mean(tf.keras.backend.square(x - x_pred), axis=[1,2,3])

        return tf.reshape(losses, (-1, 1))
