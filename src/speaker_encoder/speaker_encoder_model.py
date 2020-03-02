import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers
from speaker_encoder import resnet_model


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


class LDE(layers.Layer):
    def __init__(self, dict_size, with_bias=False, distance_type="norm", pooling="mean", **kwargs):
        super(LDE, self).__init__(**kwargs)
        self.dict_size = dict_size
        self.smoothing_factor = tf.Variable(tf.ones([dict_size]), name="smoothing factors")
        if with_bias:
            self.bias = tf.Variable(tf.zeros([dict_size]), name="bias")
        else:
            self.bias = 0
        assert distance_type == "norm" or distance_type == "sqr"
        if distance_type == "norm":
            self.dis = lambda x: tf.norm(x, axis=-1, ord=2)
        else:
            self.dis = lambda x: tf.math.reduce_sum(tf.math.square(x), axis=-1)
        self.norm = lambda x: tf.nn.softmax(-self.dis(x) * (self.smoothing_factor**2) + self.bias, axis=-2)
        assert pooling == "mean" or pooling = "mean+std"
        self.pool = pooling

    def build(self, input_shape):
       _, time_length, filters = input_shape
       self.dict = tf.Variable(tf.random.normal([self.dict_size, filters]), name="dictionary components")

    def call(self, x):
        r = tf.expand_dims(x, 2) - self.dict
        w = tf.expand_dims(self.norm(r), 3)
        w = w / (tf.math.reduce_sum(w, axis=1, keepdim=True) + 1e-9)
        if self.pool = "mean":
            x = tf.math.reduce_sum(w * r, axis=1)
        else:
            x1 = tf.math.reduce_sum(w * r, axis=1)
            x2 = tf.math.sqrt(tf.math.reduce_sum(w * r ** 2, axis=1) + 1e-8)
            x = tf.concat([x1, x2], axis=-1)
        return tf.reshape(x, [tf.shape(x).numpy()[0], -1])


class SpeakerEmbedding(tf.Model):
    def __init__(**kwargs):
        super(SpeakerEmbedding, self).__init__(**kwargs)

        # Define the encoding model for each frame
        self.layers = get_LSTM_layers()

        self.frame_encoding = resnet_model.resnet34()
        self.pool = LDE(64)

    def call(self, inputs):
        frames = self.frame_encoding(inputs)
        encoding = self.pool(frames)

        return encoding
