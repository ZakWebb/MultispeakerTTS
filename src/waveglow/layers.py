import tensorflow as tf
from tf.keras import layers
from waveglow import utils


class WaveGlowLoss(layers.Layer):
    def __init__(self, sigma=1.0, **kwargs):
        super(WaveGlowLoss, self).__init__(**kwargs)
        self.sigma = sigma

    def call(self, model_output):
        z, log_s_list, log_det_W_list = model_output

        log_det_W_total = 0
        for log_det_w in log_det_W_list:
            log_det_W_total += log_det_W

        log_s_total = 0
        for log_s in log_s_list:
            log_s_total += tf.reduce_sum(log_s)

        loss = tf.reduce_sum(z * z) / (2 * self.sigma * self.sigma) \
            - log_s_total - log_det_W_list
        return loss / tf.size(z)  # double check to make sure that z is shape (a, b, c)


class Invertable1x1Conv(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(Invertable1x1Conv, self).__init__(**kwargs)
        self.conv = layers.Conv1D(filters, kernel_size=1, stride=1, padding='same',
                                  kernel_initializer=utils.PosDetOrthogonal, bias=False)

    def call(self, input, reverse=False):

        # shape
        [batch_size, group_size, n_of_groups] = tf.shape(input)

        w = self.conv.get_weights()[0]

        if reverse:
            # do the reverse 1x1 conv

        else:
            log_det_W = batch_size * n_of_groups * tf.linalg.logdet(self.w)
