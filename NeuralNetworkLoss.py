import tensorflow as tf


class NeuralNetworkLoss:
    @staticmethod
    def meansquared(output, target, batch_size, dtype, **kwargs):
        return tf.reduce_sum(0.5 * tf.pow(output-target, 2)) / tf.constant([batch_size], dtype=dtype)

    @staticmethod
    def crossentropy(output, target, batch_size, dtype, **kwargs):
        return -tf.reduce_sum(target * tf.log(tf.clip_by_value(output, 1e-8, tf.reduce_max(output))),
                              axis=kwargs.get('axis', 0)) / \
               tf.constant([batch_size], dtype=dtype)

    # This loss is designed for the Actor Critic reinforcement learning model.
    @staticmethod
    def tdsquared(output, target, reward, gamma, **kwargs):
        td_error = reward + gamma * target - output
        return tf.square(td_error)
