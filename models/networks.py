import sys
assert sys.version_info >= (3, 0), "Python 3.0 or greater required"
from math import log

import logging
logger   = logging.getLogger(__name__)
debug    = logger.debug
info     = logger.info
warning  = logger.warning
error    = logger.error
critical = logger.critical

import tensorflow as tf

from utils.model import print_model, print_layer

class BaseModel(tf.keras.Model):
    # pylint: disable=no-member
    def call(self, inputs):
        return self.model(inputs)

    def summary(self, **kwargs):
        print_model(self.model, print_fn=kwargs.get('print_fn') or print)
        self.model.summary(**kwargs)


kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)  # Conv
beta_initializer = tf.keras.initializers.Zeros()                                # BatchNorm
gamma_initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)   # BatchNorm

class Encoder(BaseModel):
    """DCGAN Decoder Network

    Args:
        input_shape (tuple): shape of one input datum (without batch size)
        latent_size (int, optional): Size of the decoder input or of the latent space. Defaults to 100.
        n_filters (int, optional): Filter count of the initial convolution layer. Defaults to 64.
        n_extra_layers (int, optional): Count of additional layers. Defaults to 0.

    Raises:
        ValueError: If the image widht and height aren't the same. (image != quadratic)
        ValueError: If the image widht or height aren't be a power of two.
    """
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, full_dcgan_encoder=False, **kwargs):
        super().__init__(**kwargs)
        if input_shape[0] != input_shape[1]:
            raise ValueError("image width and height must be the same size")
        if log(input_shape[0], 2) != int(log(input_shape[0], 2)):
            raise ValueError("image width and height must be a power of 2")

        encoder = tf.keras.Sequential(name=kwargs.get('name') or 'encoder')

        encoder.add(tf.keras.Input(shape=input_shape, name='input_1'))

        encoder.add(tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(4,4),
            strides=(2,2),
            padding='same',
            kernel_initializer=kernel_initializer,
            use_bias=False,
            name='initial-conv-{}-{}'.format(input_shape[2], n_filters)
        ))
        encoder.add(tf.keras.layers.LeakyReLU(
            alpha=0.2,
            name='initial-relu-{}'.format(n_filters)
        ))

        last_layer_output_height = input_shape[0] // 2
        last_layer_output_depth = n_filters

        for t in range(n_extra_layers):
            encoder.add(tf.keras.layers.Conv2D(
                filters=last_layer_output_depth,
                kernel_size=(3,3),
                strides=(1,1),
                padding='same',
                kernel_initializer=kernel_initializer,
                use_bias=False,
                name='extra-conv-{}-{}'.format(t, last_layer_output_depth)
            ))
            encoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                name='extra-batchnorm-{}-{}'.format(t, last_layer_output_depth)
            ))
            encoder.add(tf.keras.layers.LeakyReLU(
                alpha=0.2,
                name='extra-relu-{}-{}'.format(t, last_layer_output_depth)
            ))

        while last_layer_output_height > 4:
            layer_output_depth = last_layer_output_depth * 2
            encoder.add(tf.keras.layers.Conv2D(
                filters=layer_output_depth,
                kernel_size=(4,4),
                strides=(2,2),
                padding='same',
                kernel_initializer=kernel_initializer,
                use_bias=False,
                name='pyramid-conv-{}-{}'.format(last_layer_output_depth, layer_output_depth)
            ))
            last_layer_output_height = last_layer_output_height / 2
            last_layer_output_depth = layer_output_depth
            encoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                name='pyramid-batchnorm-{}'.format(layer_output_depth)
            ))
            encoder.add(tf.keras.layers.LeakyReLU(
                alpha=0.2,
                name='pyramid-relu-{}'.format(layer_output_depth)
            ))

        encoder.add(tf.keras.layers.Conv2D(
            filters=latent_size,
            kernel_size=(4,4),
            strides=(1,1),
            padding='valid',
            kernel_initializer=kernel_initializer,
            use_bias=False,
            name='final-conv-{}-{}'.format(last_layer_output_depth, latent_size)
        ))
        if full_dcgan_encoder:
            encoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                name='final-batchnorm-{}'.format(latent_size)
            ))
            encoder.add(tf.keras.layers.LeakyReLU(
                alpha=0.2,
                name='final-relu-{}'.format(latent_size)
            ))

        self.model = encoder


class Decoder(BaseModel):
    """DCGAN Decoder Network

    Args:
        input_shape (tuple): shape of one input datum (without batch size)
        latent_size (int, optional): Size of the decoder input or of the latent space. Defaults to 100.
        n_filters (int, optional): Filter count of the initial convolution layer. Defaults to 64.
        n_extra_layers (int, optional): Count of additional layers. Defaults to 0.

    Raises:
        ValueError: If the image widht and height aren't the same. (image != quadratic)
        ValueError: If the image widht or height aren't be a power of two.
    """
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, **kwargs):
        super().__init__(**kwargs)
        if input_shape[0] != input_shape[1]:
            raise ValueError("image width and height must be the same size")
        if log(input_shape[0], 2) != int(log(input_shape[0], 2)):
            raise ValueError("image width and height must be a power of 2")

        cngf, tisize = n_filters // 2, 4
        while tisize != input_shape[0]:
            cngf = cngf * 2
            tisize = tisize * 2

        decoder = tf.keras.Sequential(name=kwargs.get('name') or 'decoder')

        decoder.add(tf.keras.Input(shape=(1,1,latent_size), name='input_1'))

        decoder.add(tf.keras.layers.Conv2DTranspose(
            filters=cngf,
            kernel_size=(4,4),
            strides=(1,1),
            padding='valid',
            kernel_initializer=kernel_initializer,
            use_bias=False,
            name='initial-convt-{}-{}'.format(latent_size, cngf)
        ))
        decoder.add(tf.keras.layers.BatchNormalization(
            axis=-1,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            name='initial-batchnorm-{}'.format(cngf)
        ))
        decoder.add(tf.keras.layers.ReLU(
            name='initial-relu-{}'.format(cngf)
        ))

        csize, _ = 4, cngf
        while csize < input_shape[0] // 2:
            decoder.add(tf.keras.layers.Conv2DTranspose(
                filters=cngf // 2,
                kernel_size=(4,4),
                strides=(2,2),
                padding='same',
                kernel_initializer=kernel_initializer,
                use_bias=False,
                name='pyramid-convt-{}-{}'.format(cngf, cngf // 2)
            ))
            decoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                name='pyramid-batchnorm-{}'.format(cngf // 2)
            ))
            decoder.add(tf.keras.layers.ReLU(
                name='pyramid-relu-{}'.format(cngf // 2)
            ))
            cngf = cngf // 2
            csize = csize * 2

        for t in range(n_extra_layers):
            decoder.add(tf.keras.layers.Conv2D(
                filters=cngf,
                kernel_size=(3,3),
                strides=(1,1),
                padding='same',
                kernel_initializer=kernel_initializer,
                use_bias=False,
                name='extra-conv-{}-{}'.format(t, cngf)
            ))
            decoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                name='extra-batchnorm-{}-{}'.format(t, cngf)
            ))
            decoder.add(tf.keras.layers.ReLU(
                name='extra-relu-{}-{}'.format(t, cngf)
            ))

        decoder.add(tf.keras.layers.Conv2DTranspose(
            filters=input_shape[2],
            kernel_size=(4,4),
            strides=(2,2),
            padding='same',
            kernel_initializer=kernel_initializer,
            use_bias=False,
            name='final-convt-{}-{}'.format(cngf, input_shape[2])
        ))
        decoder.add(tf.keras.layers.Activation(
            activation='tanh',
            name='final-tanh-{}'.format(input_shape[2])
        ))

        self.model = decoder