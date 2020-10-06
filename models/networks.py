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
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, **kwargs):
        super().__init__(**kwargs)
        assert input_shape[0] == input_shape[1], "image width and height must be same size"
        assert input_shape[0] % 16 == 0, "image size has to be a multiple of 16 pixel"

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

        csize, cndf = input_shape[0] / 2, n_filters

        for t in range(n_extra_layers):
            encoder.add(tf.keras.layers.Conv2D(
                filters=cndf,
                kernel_size=(3,3),
                strides=(1,1),
                padding='same',
                kernel_initializer=kernel_initializer,
                use_bias=False,
                name='extra-conv-{}-{}'.format(t, cndf)
            ))
            encoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                name='extra-batchnorm-{}-{}'.format(t, cndf)
            ))
            encoder.add(tf.keras.layers.LeakyReLU(
                alpha=0.2,
                name='extra-relu-{}-{}'.format(t, cndf)
            ))

        while csize > 4:
            old_cndf = cndf
            cndf = cndf * 2
            csize = csize / 2
            encoder.add(tf.keras.layers.Conv2D(
                filters=cndf,
                kernel_size=(4,4),
                strides=(2,2),
                padding='same',
                kernel_initializer=kernel_initializer,
                use_bias=False,
                name='pyramid-conv-{}-{}'.format(old_cndf, cndf)
            ))
            encoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                name='pyramid-batchnorm-{}'.format(cndf)
            ))
            encoder.add(tf.keras.layers.LeakyReLU(
                alpha=0.2,
                name='pyramid-relu-{}'.format(cndf)
            ))

        encoder.add(tf.keras.layers.Conv2D(
            filters=latent_size,
            kernel_size=(4,4),
            strides=(1,1),
            padding='valid',
            kernel_initializer=kernel_initializer,
            use_bias=False,
            name='final-conv-{}-{}'.format(cndf, latent_size)
        ))

        self.model = encoder


class Decoder(BaseModel):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, **kwargs):
        super().__init__(**kwargs)
        assert input_shape[0] == input_shape[1], "image width and height must be same size"
        assert input_shape[0] % 16 == 0, "image size has to be a multiple of 16 pixel"

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