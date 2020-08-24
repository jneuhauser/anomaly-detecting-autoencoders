import numpy as np
import tensorflow as tf

print_layer = lambda layer: print("  {:<24} inputs = {:>10} {:<20} outputs = {:>10} {:<20}".format(
    layer.name,
    np.prod(layer.input_shape[1:]),
    str(layer.input_shape[1:]),
    np.prod(layer.output_shape[1:]),
    str(layer.output_shape[1:])
))

class BaseModel(tf.keras.Model):
    def call(self, inputs):
        return self.model(inputs)

    def summary(self):
        print("{} layers:".format(self.model.name))
        for layer in self.model.layers:
            print_layer(layer)
        self.model.summary()


class Encoder(BaseModel):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0):
        super().__init__()
        assert input_shape[0] == input_shape[1], "image width and height must be same size"
        assert input_shape[0] % 16 == 0, "image size has to be a multiple of 16 pixel"

        encoder = tf.keras.Sequential(name='encoder')

        encoder.add(tf.keras.Input(shape=input_shape, name='input_1'))

        encoder.add(tf.keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=(4,4),
            strides=(2,2),
            padding='same',
            #kernel_regularizer='l2',
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
                #kernel_regularizer='l2',
                use_bias=False,
                name='extra-conv-{}-{}'.format(t, cndf)
            ))
            encoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
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
                #kernel_regularizer='l2',
                use_bias=False,
                name='pyramid-conv-{}-{}'.format(old_cndf, cndf)
            ))
            encoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
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
            #kernel_regularizer='l2',
            use_bias=False,
            name='final-conv-{}-{}'.format(cndf, latent_size)
        ))

        self.model = encoder


class Decoder(BaseModel):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0):
        super().__init__()
        assert input_shape[0] == input_shape[1], "image width and height must be same size"
        assert input_shape[0] % 16 == 0, "image size has to be a multiple of 16 pixel"

        cngf, tisize = n_filters // 2, 4
        #print('cngf', cngf, 'tisize', tisize)
        while tisize != input_shape[0]:
            cngf = cngf * 2
            tisize = tisize * 2
            #print('cngf', cngf, 'tisize', tisize)

        decoder = tf.keras.Sequential(name='decoder')

        decoder.add(tf.keras.Input(shape=(1,1,latent_size), name='input_1'))

        decoder.add(tf.keras.layers.Conv2DTranspose(
            filters=cngf,
            kernel_size=(4,4),
            strides=(1,1),
            padding='valid',
            use_bias=False,
            name='initial-convt-{}-{}'.format(latent_size, cngf)
        ))
        decoder.add(tf.keras.layers.BatchNormalization(
            axis=-1,
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
                use_bias=False,
                name='pyramid-convt-{}-{}'.format(cngf, cngf // 2)
            ))
            decoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
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
                use_bias=False,
                name='extra-conv-{}-{}'.format(t, cngf)
            ))
            decoder.add(tf.keras.layers.BatchNormalization(
                axis=-1,
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
            name='final-convt-{}-{}'.format(cngf, input_shape[2])
        ))
        decoder.add(tf.keras.layers.Activation(
            activation='tanh',
            name='final-tanh-{}'.format(input_shape[2])
        ))

        self.model = decoder


class NetD(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0):
        super().__init__()
        model = Encoder(input_shape, 1, n_filters, n_extra_layers).model
        layers = list(model.layers)

        self.features = tf.keras.Sequential(layers[:-1])
        self.classifier = tf.keras.Sequential(layers[-1])
        self.classifier.add(tf.keras.layers.Activation('sigmoid'))

    def call(self, x):
        features = self.features(x)
        classifier = self.classifier(features)
        #classifier = classifier.view(-1, 1).squeeze(1) # From pytorch impl
        # TODO: Is the following equivalent? Do we need this? What about batch size?
        classifier = tf.reshape(classifier, (-1, 1))
        classifier = tf.squeeze(classifier, 1)

        return classifier, features

    def summary(self):
        print("features layers:")
        for layer in self.features.layers:
            print_layer(layer)
        self.features.summary()
        print("classifier layers:")
        for layer in self.classifier.layers:
            print_layer(layer)
        self.classifier.summary()


class NetG(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0):
        super().__init__()
        self.encoder_i = Encoder(input_shape, latent_size, n_filters, n_extra_layers).model
        self.decoder = Decoder(input_shape, latent_size, n_filters, n_extra_layers).model
        self.encoder_o = Encoder(input_shape, latent_size, n_filters, n_extra_layers).model

    def call(self, x):
        latent_i = self.encoder_i(x)
        gen_img = self.decoder(latent_i)
        latent_o = self.encoder_o(gen_img)
        return gen_img, latent_i, latent_o

    def summary(self):
        print("encoder_i layers:")
        for layer in self.encoder_i.layers:
            print_layer(layer)
        self.encoder_i.summary()
        print("decoder layers:")
        for layer in self.decoder.layers:
            print_layer(layer)
        self.decoder.summary()
        print("encoder_o layers:")
        for layer in self.encoder_o.layers:
            print_layer(layer)
        self.encoder_o.summary()
