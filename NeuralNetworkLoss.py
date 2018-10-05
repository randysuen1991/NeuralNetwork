import tensorflow as tf


class NeuralNetworkLoss:
    @staticmethod
    def meansquared(output, target, batch_size):
        return tf.reduce_sum(0.5*tf.pow(output-target, 2)) / tf.constant([batch_size], dtype=tf.float64)

    @staticmethod
    def crossentropy(output, target, batch_size):
        return -tf.reduce_sum(target * tf.log(output)) / tf.constant([batch_size], dtype=tf.float64)
