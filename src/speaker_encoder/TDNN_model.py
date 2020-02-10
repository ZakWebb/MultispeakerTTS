
import tensorflow as tf
from tf.keras import layers


class TDNNBlock(layers.Layer):
    def __init__(self, **kwargs):
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
