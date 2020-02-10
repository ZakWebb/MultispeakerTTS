import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers


hidden_layer_dim = 768
projection_layer_dim = 256
num_LSTM_layers = 3


def get_model():
    tf.keras.backend.set_floatx('float64')

    return SpeakerEmbedding()


def get_LSTM_layers():
    speaker_encoder_model = []
    for layer in range(num_LSTM_layers):
        speaker_encoder_model.append(layers.LSTM(hidden_layer_dim,
                                                 name='LSTM %d' % (layer),
                                                 return_sequences=not (layer == num_LSTM_layers - 1)))
        speaker_encoder_model.append(layers.Dense(projection_layer_dim, name='Projection %d' % (layer)))
    return speaker_encoder_model


class SpeakerEmbedding(tf.Model):
    def __init__(**kwargs):
        super(SpeakerEmbedding, self).__init__(**kwargs)

        # Define the encoding model for each frame
        self.layers = get_LSTM_layers()

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x
