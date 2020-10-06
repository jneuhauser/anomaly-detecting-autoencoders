# standard modules
import os
import logging
from packaging import version

logger   = logging.getLogger('models.ganomaly')
debug    = logger.debug
info     = logger.info
warning  = logger.warning
error    = logger.error
critical = logger.critical

# external modules
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
assert version.parse('2.3') <= version.parse(tf.version.VERSION), "Tensorflow 2.3 or geater required"

def print_layer(layer, print_fn=print):
    if isinstance(layer, tf.keras.Model):
        print_model(layer)
    if not isinstance(layer, tf.keras.layers.Layer):
        raise ValueError("layer isn't a instance of tf.keras.layers.Layer")
    print_fn("  {:<24} inputs = {:>10} {:<20} outputs = {:>10} {:<20}".format(
        layer.name,
        np.prod(layer.input_shape[1:]),
        str(layer.input_shape[1:]),
        np.prod(layer.output_shape[1:]),
        str(layer.output_shape[1:])
    ))

def print_model(model, print_fn=print):
    if not isinstance(model, tf.keras.Model):
        raise ValueError("model isn't a instance of tf.keras.Model")
    print_fn('Model: "{}"'.format(model.name))
    for layer in model.layers:
        print_layer(layer, print_fn=print_fn)

kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)  # Conv
beta_initializer = tf.keras.initializers.Zeros()                                # BatchNorm
gamma_initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)   # BatchNorm

def reset_weights(model):
    # https://github.com/keras-team/keras/issues/341#issuecomment-539198392
    debug('Re-initialize weights of model: {}'.format(model.name))
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        debug('Re-initialize weights of layer: {}'.format(layer.name))
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

    def summary(self, **kwargs):
        print_model(self.model)
        self.model.summary(**kwargs)


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
        while tisize != input_shape[0]:
            cngf = cngf * 2
            tisize = tisize * 2

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


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0):
        super().__init__()
        model = Encoder(input_shape, 1, n_filters, n_extra_layers).model
        layers = list(model.layers)

        self.features = tf.keras.Sequential(layers[:-1])
        self.classifier = tf.keras.Sequential(layers[-1])
        self.classifier.add(tf.keras.layers.Reshape((1,)))
        self.classifier.add(tf.keras.layers.Activation('sigmoid'))

    def call(self, x, training=False):
        features = self.features(x, training)
        classifier = self.classifier(features, training)

        return classifier, features

    def summary(self, **kwargs):
        #print_model(self)
        super().summary(**kwargs)
        super().features.summary(**kwargs)
        super().classifier.summary(**kwargs)


class Generator(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0):
        super().__init__()
        self.encoder_i = Encoder(input_shape, latent_size, n_filters, n_extra_layers).model
        self.decoder = Decoder(input_shape, latent_size, n_filters, n_extra_layers).model
        self.encoder_o = Encoder(input_shape, latent_size, n_filters, n_extra_layers).model

    def call(self, x, training=False):
        latent_i = self.encoder_i(x, training)
        fake = self.decoder(latent_i, training)
        latent_o = self.encoder_o(fake, training)
        return fake, latent_i, latent_o

    def summary(self, **kwargs):
        #print_model(self)
        super().summary(**kwargs)
        self.encoder_i.summary(**kwargs)
        self.decoder.summary(**kwargs)
        self.encoder_o.summary(**kwargs)


class GANomaly(tf.keras.Model):
    def __init__(self, input_shape, latent_size=100, n_filters=64, n_extra_layers=0, resume=False, resume_path=None):
        super().__init__()

        self.net_gen = Generator(input_shape, latent_size, n_filters, n_extra_layers)
        self.net_dis = Discriminator(input_shape, latent_size, n_filters, n_extra_layers)

        # resume from stored weights
        if resume:
            self.load_weights(resume_path)
        self.resume_path = resume_path

        # input TODO Do we need any of this? Probably fixed_input...
        #self.input = None
        #self.label = None
        #self.ground_truth = None
        #self.fixed_input = None

    def compile(self,
        loss: dict=dict(),
        loss_weights: dict=dict(),
        optimizer: dict=dict(),
        **kwargs
    ):
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
        # test_step():  https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/training.py#L1148-L1180
        # evaluate():   https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/training.py#L1243-L1394
        # fit():        https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/engine/training.py#L824-L1146

        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        # x.shape: (batchsize, width, height, depth)
        # y.shape: (batchsize, 1) on numpy array or (batchsize,) on tf.data.Dataset

        _, latent_i, latent_o = self.net_gen(x, training=False)
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

        _, latent_i, latent_o = self.net_gen(x, training=False)
        # letent_x.shape: (batchsize, 1, 1, latent_size)

        error = tf.keras.backend.mean(tf.keras.backend.square(latent_i - latent_o), axis=-1)
        # error.shape: (batchsize, 1, 1, 1)

        return tf.reshape(error, (-1, 1))


class ADModelEvaluator(tf.keras.callbacks.Callback):
    def __init__(self, test_count, model_dir=None):
        super().__init__()

        self.model_dir = model_dir

        # AUROC as fixed metric
        self.metric = tf.keras.metrics.AUC(
            curve='ROC',
            summation_method='interpolation'
        )

        self.test_labels = np.zeros((test_count, 1), dtype=np.float32)
        self.test_losses = np.zeros((test_count, 1), dtype=np.float32)
        self.test_results_idx_start = 0
        self.test_results_idx_end = 0

        self.test_ptp_loss = 0.
        self.test_min_loss = 0.
        self.test_result = 0.
        self.test_results = []

        self.best_ptp_loss = 0.
        self.best_min_loss = 0.
        self.best_result = 0.
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        # Save epoch result history
        self.test_results.append(self.test_result)
        # Keep track of best metric and save best model
        if self.test_result >= self.best_result:
            self.best_epoch = epoch
            self.best_result = self.test_result
            self.best_ptp_loss = self.test_ptp_loss
            self.best_min_loss = self.test_min_loss
            self.best_weights = self.model.get_weights()
            if self.model_dir:
                self.model.save_weights(path=self.model_dir)
        # Print determined values
        info("Curr Epoch {:02d}: AUC(ROC): {:.5f}, ptp_loss: {:.5f}, min_loss: {:.5f}".format(
            epoch+1,
            self.test_result,
            self.test_ptp_loss,
            self.test_min_loss
        ))
        info("Best Epoch {:02d}: AUC(ROC): {:.5f}, ptp_loss: {:.5f}, min_loss: {:.5f}".format(
            self.best_epoch+1,
            self.best_result,
            self.best_ptp_loss,
            self.best_min_loss
        ))
        #debug("TP: {}\nTN: {}\nFP: {}\nFN: {}\nTH: {}".format(
        #    self.metric.true_positives,
        #    self.metric.true_negatives,
        #    self.metric.false_positives,
        #    self.metric.false_negatives,
        #    self.metric.thresholds
        #))

    def on_test_begin(self, logs=None):
        # Prepare for new evaluation
        self.metric.reset_states()
        self.test_results_idx_start = 0
        self.test_results_idx_end = 0

    def on_test_end(self, logs=None):
        # Raise error if not enoght results are collected
        if self.test_results_idx_end != self.test_losses.shape[0]:
            raise ValueError("collected results count of {} doesn't match expected count of {}"
                .format(self.test_results_idx_end, self.test_losses.shape[0]))
        # Scale losses between [0, 1]
        self.test_min_loss = np.min(self.test_losses)
        self.test_ptp_loss = np.max(self.test_losses) - self.test_min_loss
        self.test_losses -= self.test_min_loss
        self.test_losses /= self.test_ptp_loss
        # Calculate metric AUROC
        self.metric.update_state(self.test_labels, self.test_losses)
        self.test_result = self.metric.result().numpy()

    def on_test_batch_end(self, batch, logs=None):
        # Gather all per batch losses and labels
        labels = logs.get('labels')
        losses = logs.get('losses')
        batch_size = losses.shape[0]
        self.test_results_idx_end += batch_size
        self.test_labels[self.test_results_idx_start:self.test_results_idx_end] = labels
        self.test_losses[self.test_results_idx_start:self.test_results_idx_end] = losses
        self.test_results_idx_start += batch_size
