import numpy as np
import tensorflow as tf

class AE(tf.keras.Model):
    """Parent class for generic autoencoders"""
    def __init__(self, encoder, decoder, **kwargs):
        super(AE, self).__init__(**kwargs)
        if not self._is_model(encoder) or not self._is_model(decoder):
            raise ValueError("encoder or decoder is not a valid model class")
        self.encoder = encoder
        self.decoder = decoder
    
    @staticmethod
    def _is_model(object):
        return (
            issubclass(type(object), tf.keras.Model) or
            issubclass(type(object), tf.keras.Sequential)
        )
    
    def call(self, inputs):
        """Connect encoder and decoder for forward pass"""
        x = self.encoder(inputs)
        return self.decoder(x)
    
    def print_summary(self):
        """Print summary about encoder and decoder"""
        print_layer = lambda layer: print("  {:<24} inputs = {:>10} {:<20} outputs = {:>10} {:<20}".format(
            layer.name,
            np.prod(layer.input_shape[1:]),
            str(layer.input_shape[1:]),
            np.prod(layer.output_shape[1:]),
            str(layer.output_shape[1:])
        ))
        print("Encoder layers:")
        self.encoder.summary()
        #for layer in self.encoder.layers:
        #    print_layer(layer)
        print("Decoder layers:")
        self.decoder.summary()
        #for layer in self.decoder.layers:
        #    print_layer(layer)
        
        print("Autoencoder summary:")
        print("  Input values:      {} {}".format(
            np.prod(self.encoder.layers[0].input_shape[1:]),
            self.encoder.layers[0].input_shape[1:]
        ))
        print("  Latent space:      {} {}".format(
            np.prod(self.decoder.layers[0].input_shape[1:]),
            self.decoder.layers[0].input_shape[1:]
        ))
        print("  Output values:     {} {}".format(
            np.prod(self.decoder.layers[-1].output_shape[1:]),
            self.decoder.layers[-1].output_shape[1:]
        ))
        print("  Compression ratio: {}:1".format(
            np.prod(self.encoder.layers[0].input_shape[1:]) / np.prod(self.decoder.layers[0].input_shape[1:])
        ))
