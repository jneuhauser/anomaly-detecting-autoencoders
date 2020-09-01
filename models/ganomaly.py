import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
import os

def print_layer(layer):
    if isinstance(layer, tf.keras.Model):
        print_model(layer)
    if not isinstance(layer, tf.keras.layers.Layer):
        raise ValueError("layer isn't a instance of tf.keras.layers.Layer")
    print("  {:<24} inputs = {:>10} {:<20} outputs = {:>10} {:<20}".format(
        layer.name,
        np.prod(layer.input_shape[1:]),
        str(layer.input_shape[1:]),
        np.prod(layer.output_shape[1:]),
        str(layer.output_shape[1:])
    ))

def print_model(model):
    if not isinstance(model, tf.keras.Model):
        raise ValueError("model isn't a instance of tf.keras.Model")
    print('Model: "{}"'.format(model.name))
    for layer in model.layers:
        print_layer(layer)

kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)  # Conv
beta_initializer = tf.keras.initializers.Zeros()                                # BatchNorm
gamma_initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)   # BatchNorm

def reset_weights(model):
    # https://github.com/keras-team/keras/issues/341#issuecomment-539198392
    print('Re-initialize weights of model: {}'.format(model.name))
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        print('Re-initialize weights of layer: {}'.format(layer.name))
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            var = getattr(layer, k.replace("_initializer", ""))
            if var is not None:
                var.assign(initializer(var.shape, var.dtype))


class BaseModel(tf.keras.Model):
    # pylint: disable=no-member
    def call(self, inputs):
        return self.model(inputs)

    def summary(self):
        print_model(self.model)
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


class NetD(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0):
        super().__init__()
        model = Encoder(input_shape, 1, n_filters, n_extra_layers).model
        layers = list(model.layers)

        self.features = tf.keras.Sequential(layers[:-1])
        self.classifier = tf.keras.Sequential(layers[-1])
        self.classifier.add(tf.keras.layers.Reshape((1,))) # (batchsize, 1, 1, 1) -> (batchsize, 1) instead of -> (batchsize,)
        self.classifier.add(tf.keras.layers.Activation('sigmoid'))

    def call(self, x, training=False):
        features = self.features(x, training)
        classifier = self.classifier(features, training)
        # From pytorch impl (batchsize, 1, 1, 1) -> (batchsize,)
        #classifier = classifier.view(-1, 1).squeeze(1)
        # Is the following equivalent? Do we need this? What about batch size?
        #  -> We are using a keras.layer.Reshape() for (1, 1, 1) -> (1,)
        #   -> Anyway, the shape is not relevant as we use tf.ones_like() and
        #      tf.zeros_like() for discriminator loss calculation.
        #classifier = tf.reshape(classifier, (-1, 1))
        #classifier = tf.squeeze(classifier, 1)

        return classifier, features

    def summary(self):
        print_model(self)
        super().summary()


class NetG(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0):
        super().__init__()
        self.encoder_i = Encoder(input_shape, latent_size, n_filters, n_extra_layers).model
        self.decoder = Decoder(input_shape, latent_size, n_filters, n_extra_layers).model
        self.encoder_o = Encoder(input_shape, latent_size, n_filters, n_extra_layers).model

    def call(self, x, training=False):
        latent_i = self.encoder_i(x, training)
        gen_img = self.decoder(latent_i, training)
        latent_o = self.encoder_o(gen_img, training)
        return latent_i, gen_img, latent_o

    def summary(self):
        print_model(self)
        super().summary()


class GANomaly(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, resume=False, resume_path=None):
        super().__init__()
        self.netg = NetG(input_shape, latent_size, n_filters, n_extra_layers)
        self.netd = NetD(input_shape, latent_size, n_filters, n_extra_layers)
        # apply weights_init
        # https://github.com/samet-akcay/ganomaly/blob/master/lib/networks.py#L11
        # https://keras.io/api/layers/initializers/
        # https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668

        # resume from stored weights
        if resume:
            self.load_weights(resume_path)
        self.resume_path = resume_path

        # losses
        self.loss_adv = tf.keras.losses.MeanSquaredError()
        self.weight_adv = 1 # TODO Make it a param
        self.loss_con = tf.keras.losses.MeanAbsoluteError()
        self.weight_con = 50 # TODO Make it a param
        self.loss_enc = tf.keras.losses.MeanSquaredError()
        self.weight_enc = 1 # TODO Make it a param
        self.loss_bce = tf.keras.losses.BinaryCrossentropy()

        # input TODO Do we need any of this? Probably fixed_input...
        #self.input = None
        #self.label = None
        #self.ground_truth = None
        #self.fixed_input = None
        #self.real_label = tf.ones(shape=(batchsize,), dtype=tf.dtypes.float32)
        #self.fake_label = tf.zeros(shape=(batchsize,), dtype=tf.dtypes.float32)

        # optimizer
        # TODO Make learing_rate and beta_1 a param
        self.optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
        self.optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

    def load_weights(self, path):
        #if os.path.exists(resume_path):
        print('Loading pre-trained network weights from: "{}"'.format(
            os.path.abspath(path)), end=' ')
        self.netg.load_weights(os.path.join(path, 'generator'))
        self.netd.load_weights(os.path.join(path, 'discriminator'))
        print("-> Done\n")

    def save_weights(self, path):
        print('Saving pre-trained network weights to: "{}"'.format(
            os.path.abspath(path)), end=' ')
        self.netg.save_weights(os.path.join(path, 'generator'))
        self.netd.save_weights(os.path.join(path, 'discriminator'))
        print("-> Done\n")

    def call(self, x):
        # TODO output of netg ???
        return self.netg(x)[1], self.netd(x)[0]

    #@tf.function(autograph=False) # disable inherited tf.function(autograph=True) decorator
    def train_step_old(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            latent_i, fake, latent_o = self.netg(data)

            err_g_adv = self.loss_adv(self.netd(data)[1], self.netd(fake)[1])
            err_g_con = self.loss_con(data, fake)
            err_g_enc = self.loss_enc(latent_i, latent_o)
            err_g = err_g_adv * self.weight_adv + \
                    err_g_con * self.weight_con + \
                    err_g_enc * self.weight_enc

        # we are only traning the trainable_weights and not all trainable_variables (TODO right assumption???)
        grads_g = tape.gradient(err_g, self.netg.trainable_weights)
        self.optimizer_g.apply_gradients(zip(grads_g, self.netg.trainable_weights))

        with tf.GradientTape() as tape:
            pred_real, _ = self.netd(data)
            pred_fake, _ = self.netd(fake)

            err_d_real = self.loss_bce(tf.ones_like(pred_real), pred_real)
            err_d_fake = self.loss_bce(tf.zeros_like(pred_fake), pred_fake)
            err_d = (err_d_real + err_d_fake) * 0.5

        # we are only traning the trainable_weights and not all trainable_variables (TODO right assumption???)
        grads_d = tape.gradient(err_d, self.netd.trainable_weights)
        self.optimizer_d.apply_gradients(zip(grads_d, self.netd.trainable_weights))

        #if err_d < 1e-5: reset_weights(self.netd)
        # OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool` is not allowed: AutoGraph did convert this function. This might indicate you are trying to use an unsupported feature.
        # Replace with: https://www.tensorflow.org/api_docs/python/tf/cond
        tf.cond(tf.less(err_d, 1e-5), true_fn=lambda: reset_weights(self.netd), false_fn=lambda: None)

        return {
            "err_g": err_g,
            "err_g_adv": err_g_adv,
            "err_g_con": err_g_con,
            "err_g_enc": err_g_enc,
            "err_d": err_d,
            "err_d_real": err_d_real,
            "err_d_fake": err_d_fake
        }

    #@tf.function(autograph=False) # disable inherited tf.function(autograph=True) decorator
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape(watch_accessed_variables=False) as tape_g, tf.GradientTape(watch_accessed_variables=False) as tape_d:
            tape_g.watch(self.netg.trainable_weights)
            tape_d.watch(self.netd.trainable_weights)

            latent_i, fake, latent_o = self.netg(data, training=True)

            pred_real, feat_real = self.netd(data, training=True)
            pred_fake, feat_fake = self.netd(fake, training=True)

            err_g_adv = self.loss_adv(feat_real, feat_fake)
            err_g_con = self.loss_con(data, fake)
            err_g_enc = self.loss_enc(latent_i, latent_o)
            err_g = err_g_adv * self.weight_adv + \
                    err_g_con * self.weight_con + \
                    err_g_enc * self.weight_enc

            err_d_real = self.loss_bce(tf.ones_like(pred_real), pred_real)
            err_d_fake = self.loss_bce(tf.zeros_like(pred_fake), pred_fake)
            err_d = (err_d_real + err_d_fake) * 0.5

        grads_g = tape_g.gradient(err_g, self.netg.trainable_weights)
        self.optimizer_g.apply_gradients(zip(grads_g, self.netg.trainable_weights))

        grads_d = tape_d.gradient(err_d, self.netd.trainable_weights)
        self.optimizer_d.apply_gradients(zip(grads_d, self.netd.trainable_weights))

        #if err_d < 1e-5: reset_weights(self.netd)
        # OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool` is not allowed: AutoGraph did convert this function. This might indicate you are trying to use an unsupported feature.
        # Replace with: https://www.tensorflow.org/api_docs/python/tf/cond
        tf.cond(tf.less(err_d, 1e-5), true_fn=lambda: reset_weights(self.netd), false_fn=lambda: None)

        return {
            "err_g": err_g,
            "err_g_adv": err_g_adv,
            "err_g_con": err_g_con,
            "err_g_enc": err_g_enc,
            "err_d": err_d,
            "err_d_real": err_d_real,
            "err_d_fake": err_d_fake
        }
