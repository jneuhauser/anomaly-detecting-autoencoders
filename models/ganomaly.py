# standard modules
import sys
assert sys.version_info >= (3, 0), "Python 3.0 or greater required"
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


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, **kwargs):
        """GANomaly Generator Model

        Args:
            input_shape (tuple): shape of one input datum (without batch size)
            latent_size (int, optional): Size of the decoder input or of the latent space. Defaults to 100.
            n_filters (int, optional): Filter count of the initial convolution layer. Defaults to 64.
            n_extra_layers (int, optional): Count of additional layers. Defaults to 0.
        """
        kwargs['name'] = type(self).__name__
        super().__init__(**kwargs)
        model = Encoder(input_shape, 1, n_filters, n_extra_layers).model
        layers = list(model.layers)

        self.features = tf.keras.Sequential(layers[:-1], name='features')
        self.classifier = tf.keras.Sequential(layers[-1], name='classifier')
        self.classifier.add(tf.keras.layers.Reshape((1,)))
        self.classifier.add(tf.keras.layers.Activation('sigmoid'))

    def summary(self, **kwargs):
        print_model(self, print_fn=kwargs.get('print_fn') or print)
        super().summary(**kwargs)
        self.features.summary(**kwargs)
        self.classifier.summary(**kwargs)

    def call(self, x, training=False):
        features = self.features(x, training)
        classifier = self.classifier(features, training)

        return classifier, features


class Generator(tf.keras.Model):
    """GANomaly Generator Model

    Args:
        input_shape (tuple): shape of one input datum (without batch size)
        latent_size (int, optional): Size of the decoder input or of the latent space. Defaults to 100.
        n_filters (int, optional): Filter count of the initial convolution layer. Defaults to 64.
        n_extra_layers (int, optional): Count of additional layers. Defaults to 0.
    """
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, **kwargs):
        kwargs['name'] = type(self).__name__
        super().__init__(**kwargs)
        self.encoder_i = Encoder(input_shape, latent_size, n_filters, n_extra_layers, name='encoder_i').model
        self.decoder = Decoder(input_shape, latent_size, n_filters, n_extra_layers, name='decoder').model
        self.encoder_o = Encoder(input_shape, latent_size, n_filters, n_extra_layers, name='encoder_o').model

    def summary(self, **kwargs):
        print_model(self, print_fn=kwargs.get('print_fn') or print)
        super().summary(**kwargs)
        self.encoder_i.summary(**kwargs)
        self.decoder.summary(**kwargs)
        self.encoder_o.summary(**kwargs)

    def call(self, x, training=False):
        latent_i = self.encoder_i(x, training)
        fake = self.decoder(latent_i, training)
        latent_o = self.encoder_o(fake, training)
        return fake, latent_i, latent_o

    def test_step(self, data):
        # test_step():  https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/training.py#L1148-L1180
        # evaluate():   https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/training.py#L1243-L1394
        # fit():        https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/training.py#L824-L1146

        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        # x.shape: (batchsize, width, height, depth)
        # y.shape: (batchsize, 1) on numpy array or (batchsize,) on tf.data.Dataset

        _, latent_i, latent_o = self(x, training=False)
        # letent_x.shape: (batchsize, 1, 1, latent_size)

        error = tf.keras.backend.mean(tf.keras.backend.square(latent_i - latent_o), axis=-1)
        # error.shape: (batchsize, 1, 1, 1)

        return {
            "losses": tf.reshape(error, (-1, 1)),
            "labels": tf.reshape(y, (-1, 1))
            }

    def predict_step(self, data):
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/training.py#L1396

        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        # x.shape: (batchsize, width, height, depth)

        _, latent_i, latent_o = self(x, training=False)
        # letent_x.shape: (batchsize, 1, 1, latent_size)

        error = tf.keras.backend.mean(tf.keras.backend.square(latent_i - latent_o), axis=-1)
        # error.shape: (batchsize, 1, 1, 1)

        return tf.reshape(error, (-1, 1))


class GANomaly(tf.keras.Model):
    """GANomaly Model

    Args:
        input_shape (tuple): shape of one input datum (without batch size)
        latent_size (int, optional): Size of the decoder input or of the latent space. Defaults to 100.
        n_filters (int, optional): Filter count of the initial convolution layer. Defaults to 64.
        n_extra_layers (int, optional): Count of additional layers. Defaults to 0.
    """
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, **kwargs):
        kwargs['name'] = type(self).__name__
        super().__init__(**kwargs)

        self.net_gen = Generator(input_shape, latent_size, n_filters, n_extra_layers)
        self.net_dis = Discriminator(input_shape, latent_size, n_filters, n_extra_layers)

    def compile(self,
        loss: dict=dict(),
        loss_weights: dict=dict(),
        optimizer: dict=dict(),
        **kwargs
    ):
        # Calling compile of the parent class does only work on tf 2.3 and greater
        assert version.parse('2.3') <= version.parse(tf.version.VERSION), "Tensorflow 2.3 or geater required"
        super().compile(**kwargs)

        self.loss_adv = loss.get('adv', tf.keras.losses.MeanSquaredError())
        self.loss_rec = loss.get('rec', tf.keras.losses.MeanAbsoluteError())
        self.loss_enc = loss.get('enc', tf.keras.losses.MeanSquaredError())
        self.loss_dis = loss.get('dis', tf.keras.losses.BinaryCrossentropy())

        self.loss_wgt_adv = float(loss_weights.get('adv', 1))
        self.loss_wgt_rec = float(loss_weights.get('rec', 50))
        self.loss_wgt_enc = float(loss_weights.get('enc', 1))

        self.optimizer_gen = optimizer.get('gen',
            tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999))
        self.optimizer_dis = optimizer.get('dis',
            tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999))

    def load_weights(self, path):
        if not (os.path.isfile(os.path.join(path, 'generator.index')) and
                os.path.isfile(os.path.join(path, 'discriminator.index'))):
            warning('No valid pre-trained network weights in: "{}"'.format(path))
            return
        self.net_gen.load_weights(os.path.join(path, 'generator'))
        self.net_dis.load_weights(os.path.join(path, 'discriminator'))
        info('Loaded pre-trained network weights from: "{}"'.format(path))

    def save_weights(self, path):
        self.net_gen.save_weights(os.path.join(path, 'generator'))
        self.net_dis.save_weights(os.path.join(path, 'discriminator'))
        info('Saved pre-trained network weights to: "{}"'.format(path))

    def call(self, x, training=False):
        # returns: reconstructed, latent_i, latent_o, classifier, features
        return self.net_gen(x, training=training), self.net_dis(x, training=training)

    # disable inherited tf.function(autograph=True) decorator
    #@tf.function(autograph=False)
    def train_step(self, data):
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/training.py#L716
        real, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape(watch_accessed_variables=False) as tape_gen, \
             tf.GradientTape(watch_accessed_variables=False) as tape_dis:
            tape_gen.watch(self.net_gen.trainable_weights)
            tape_dis.watch(self.net_dis.trainable_weights)

            fake, latent_i, latent_o = self.net_gen(real, training=True)

            real_pred, real_feat = self.net_dis(real, training=True)
            fake_pred, fake_feat = self.net_dis(fake, training=True)

            loss_gen_adv = self.loss_adv(real_feat, fake_feat)
            loss_gen_rec = self.loss_rec(real, fake)
            loss_gen_enc = self.loss_enc(latent_i, latent_o)
            loss_gen = loss_gen_adv * self.loss_wgt_adv + \
                    loss_gen_rec * self.loss_wgt_rec + \
                    loss_gen_enc * self.loss_wgt_enc

            loss_dis_real = self.loss_dis(tf.ones_like(real_pred), real_pred)
            loss_dis_fake = self.loss_dis(tf.zeros_like(fake_pred), fake_pred)
            loss_dis = (loss_dis_real + loss_dis_fake) * 0.5

        grads_gen = tape_gen.gradient(loss_gen, self.net_gen.trainable_weights)
        self.optimizer_gen.apply_gradients(zip(grads_gen, self.net_gen.trainable_weights))

        grads_dis = tape_dis.gradient(loss_dis, self.net_dis.trainable_weights)
        self.optimizer_dis.apply_gradients(zip(grads_dis, self.net_dis.trainable_weights))

        tf.cond(tf.less(loss_dis, 1e-5), true_fn=lambda: reset_weights(self.net_dis), false_fn=lambda: None)

        return {
            "loss_gen": loss_gen,
            "loss_gen_adv": loss_gen_adv,
            "loss_gen_rec": loss_gen_rec,
            "loss_gen_enc": loss_gen_enc,
            "loss_dis": loss_dis,
            "loss_dis_real": loss_dis_real,
            "loss_dis_fake": loss_dis_fake
        }

    def test_step(self, data):
        return self.net_gen.test_step(data)

    def predict_step(self, data):
        return self.net_gen.predict_step(data)
