import tensorflow as tf
import numpy as np
import os
from speaker_encoder.speaker_encoder_model import get_model
from load_data import load_speech_data

current_language = "ga-IE"


def train_model():
    speakers_train = load_speech_data.setup_speaker_data('test.tsv', create_files=False)
    speakers_test = load_speech_data.setup_speaker_data('dev.tsv', create_files=False)

    model = get_model()
    W = tf.Variable(10.0, constraint=tf.keras.constraints.NonNeg(), dtype=tf.float64)
    b = tf.Variable(-5.0, dtype=tf.float64)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1)

    epochs = 5
    batch_speaker_num = 10
    batch_speaker_utterance_num = 8
    batches_per_epoch = 1000
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch))

        for utterance_batch in \
            load_speech_data.get_encoder_utterance_batches(speakers_train, batches_per_epoch,
                                                           batch_speaker_num,
                                                           batch_speaker_utterance_num):
            with tf.GradientTape() as tape:
                encodings = model(utterance_batch, training=True)
                normalized_encodings = tf.keras.utils.normalize(encodings, axis=1)
                normalized_encodings = tf.reshape(normalized_encodings,
                                                  [batch_speaker_num, batch_speaker_utterance_num, -1])

                similarity_mat = similarity(normalized_encodings, W, b)

                loss_value = encoding_loss_fn(similarity_mat)

            trainable_vars = model.trainable_weights + [W, b]
            grads = tape.gradient(loss_value, trainable_vars)
            grads = [grad if grad is not None else tf.zeros_like(var)
                     for var, grad in zip(trainable_vars, grads)]
            optimizer.apply_gradients(zip(grads, trainable_vars))

        # Test this epoch
        for utterance_batch in \
            load_speech_data.get_encoder_utterance_batches(speakers_test, 1,
                                                           batch_speaker_num,
                                                           batch_speaker_utterance_num):

            encodings = model(utterance_batch, training=True)
            normalized_encodings = tf.keras.utils.normalize(encodings, axis=1)
            normalized_encodings = tf.reshape(normalized_encodings,
                                              [batch_speaker_num, batch_speaker_utterance_num, -1])

            similarity_mat = similarity(normalized_encodings, W, b)

            loss_value = encoding_loss_fn(similarity_mat)
            print("Similarity matrix at end of epoch %d:" % epoch)
            print(similarity_mat)
            print("Loss for test: %f" % loss_value.numpy())


def similarity(encodings, W, b):
    num_speakers, num_utterances, _ = encodings.shape
    to_centroids = np.ones([num_utterances, num_utterances + 1])
    for i in range(num_utterances):
        to_centroids[i, i] = 0.0
    centroids = tf.tensordot(to_centroids, encodings, axes=[[0], [1]])
    centroids = tf.keras.utils.normalize(centroids, axis=2)
    similarity = np.zeros((num_speakers, num_speakers, num_utterances))
    for i in range(num_speakers):
        for j in range(num_speakers):
            for k in range(num_utterances):
                if i == j:
                    similarity[i, j, k] = tf.tensordot(centroids[k, i], encodings[j, k], 1)
                else:
                    similarity[i, j, k] = tf.tensordot(centroids[num_utterances, i],
                                                       encodings[j, k], 1)
    return similarity * W + b


def encoding_loss_fn(similarity):
    _, num_speakers, num_utterances = similarity.shape

    loss = 0.0

    for i in range(num_speakers):
        for j in range(num_utterances):
            loss += single_encoding_loss(similarity[::, i, j], i)

    return loss


def single_encoding_loss(similarities, speaker):
    return tf.math.log(tf.reduce_sum(tf.math.exp(similarities))) - similarities[speaker]


if __name__ == '__main__':
    train_model()
