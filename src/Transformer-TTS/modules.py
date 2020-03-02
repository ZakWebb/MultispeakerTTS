import tensorflow as tf
from tensorflow.keras import layers
import fastspeech.layers


def pre_net(hidden_dim, output_dim, dropout_rate=0.1):
    pre_net_layers = [
        layers.Dense(hidden_dim,
                     kernel_initializer=tf.keras.initializers.GlorotUniform,
                     activation=tf.keras.activations.relu),
        layers.Dropout(dropout_rate),
        layers.Dense(output_dim,
                     kernel_initializer=tf.keras.initializers.GlorotUniform,
                     activation=tf.keras.activations.relu),
        layers.Dropout(dropout_rate)
    ]

    return layers.Sequential(pre_net_layers)


def post_net(n_mel_channels=80, post_net_embedding_dim=512, post_net_kernel_size=5,
             postnet_n_convolutions=5, dropout_rate=0.1):
    post_net_layers = []

    for _ in range(postnet_n_convolutions - 1):
        post_net_layers += [
            layers.Conv1D(post_net_embedding_dim, post_net_kernel_size, padding="same",
                          kernel_initializer=tf.keras.initializers.GlorotUniform),
            layers.BatchNormalization(activation=tf.keras.activation.tanh),
            layers.Dropout(dropout_rate)]

    post_net_layers += [
        layers.Conv1D(n_mel_channels, post_net_kernel_size, padding="same",
                      kernel_initializer=tf.keras.initializers.GlorotUniform),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate)]

    return layers.Sequential(post_net_layers)


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
