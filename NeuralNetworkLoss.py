import tensorflow as tf


class NeuralNetworkLoss():
    def MeanSqaured(output,target,batch_size):
        return tf.reduce_sum(0.5*tf.pow(output-target,2)) / tf.constant([batch_size],dtype=tf.float64) 
    
    def CrossEntropy(output,target,batch_size):
        return tf.reduce_sum(target * tf.log(output)) / tf.constant([batch_size],dtype=tf.float32)