import logging
logger   = logging.getLogger(__name__)
debug    = logger.debug
info     = logger.info
warning  = logger.warning
error    = logger.error
critical = logger.critical

import tensorflow as tf
import numpy as np


def print_layer(layer, print_fn=print):
    if isinstance(layer, tf.keras.Model):
        print_model(layer, print_fn=print_fn)
    if not isinstance(layer, tf.keras.layers.Layer):
        raise ValueError("layer isn't a instance of tf.keras.layers.Layer")
    print_fn("  {:<24} inputs = {:>10} {:<20} outputs = {:>10} {:<20}".format(
        layer.name,
        np.prod(layer.input_shape[1:] if isinstance(layer.input_shape, tuple) else 0),
        str(layer.input_shape),
        np.prod(layer.output_shape[1:] if isinstance(layer.output_shape, tuple) else 0),
        str(layer.output_shape),
        #layer.count_params()
    ))

def print_model(model, print_fn=print):
    if not isinstance(model, tf.keras.Model):
        raise ValueError("model isn't a instance of tf.keras.Model")
    print_fn("Model: {:<19} params = {:>10}".format(
        model.name,
        model.count_params()
    ))
    for layer in model.layers:
        print_layer(layer, print_fn=print_fn)

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
