import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers


hidden_layer_dim = 768
projection_layer_dim = 256
num_LSTM_layers = 3


def get_model():
    tf.keras.backend.set_floatx('float64')
    speaker_encoder_model = tf.keras.Sequential()
    for layer in range(num_LSTM_layers):
        speaker_encoder_model.add(layers.LSTM(hidden_layer_dim,
                                              name='LSTM %d' % (layer),
                                              return_sequences=not (layer == num_LSTM_layers - 1)))
        speaker_encoder_model.add(layers.Dense(projection_layer_dim, name='Projection %d' % (layer)))

    return speaker_encoder_model


class TDNNBlock(layers.Layer):
    def __init__(self):
        super(TDNNBlock, self, **kwargs).__init__(**kwargs)

        tdnn_schedule = [(5, 1), (3, 2), (3, 3)]
        self.conv_layers = []
        for size, dilation in tdnn_schedule:
            self.conv_layers.append(layers.Conv1D(512, size, dilation=dilation, activation='relu', use_bias=False))

        self.dropout_rate = 0.15
        self.dropout = layers.Dropout(rate=self.dropout_rate)

    def call(self, inputs, training=False, ):
        x = inputs
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            if training:
                x = self.dropout(x)

        return x
