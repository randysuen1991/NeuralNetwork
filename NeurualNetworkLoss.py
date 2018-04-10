import tensorflow as tf


class NeuralNetworkLoss():
    def MeanSqaure(output,target,batch_size):
        return tf.reduce_sum(0.5*tf.pow(output-target,2))/ tf.constant([batch_size],dtype=tf.float64) 