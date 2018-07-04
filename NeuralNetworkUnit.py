import tensorflow as tf
import numpy as np


class NeuralNetworkUnit:
    def __init__(self, hidden_dim, input_dim, transfer_fun, dtype=tf.float64):
        self.input = None
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.transfer_fun = transfer_fun
        self.parameters = dict()


class NeuronLayer(NeuralNetworkUnit):
    def __init__(self, hidden_dim, input_dim=None, transfer_fun=tf.sigmoid, dtype=tf.float64):
        super().__init__(hidden_dim, input_dim, transfer_fun, dtype=dtype)

    def Initialize(self, input_dim):
        self.input_dim = input_dim
        self.parameters['w'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,
                                                                             shape=(self.input_dim, self.hidden_dim),
                                                                             mean=0,
                                                                             stddev=0.1))

        self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,
                                                                             shape=(1, self.hidden_dim),
                                                                             mean=0,
                                                                             stddev=0.1))
        self.output = tf.matmul(self.input, self.parameters['w']) + self.parameters['b']
        
        # When I don't want the result to be transformed. I will pass None to the transfer_fun.
        try:
            self.output = self.transfer_fun(self.output)
        except TypeError:
            self.output = self.output


class SoftMaxLayer:
    def Initialize(self, *args):
        sum_exp = tf.reduce_sum(tf.exp(self.input), axis=1)
        sum_exp = tf.expand_dims(sum_exp, axis=1)
        self.output = tf.divide(tf.exp(self.input), sum_exp)


class ConvolutionUnit(NeuralNetworkUnit):
    # The shape parameter should be (height, width, num filters)
    def __init__(self, shape, transfer_fun=None, dtype=tf.float64, **kwargs):
        super().__init__(None, None, transfer_fun)
        self.dtype = dtype
        self.shape = shape
        self.kwargs = kwargs
    def Initialize(self, num_channels):
        shape = list(self.shape)
        shape.insert(2, num_channels)
        shape = tuple(shape)
        self.parameters['w'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=shape,mean=0,stddev=0.1))
        self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(shape[-1],),mean=0,stddev=0.1))
        self.output = tf.nn.conv2d(self.input,self.parameters['w'], strides = self.kwargs.get('strides', [1, 1, 1, 1]),
                                   padding=self.kwargs.get('padding', 'SAME'))
        self.output = self.output + self.parameters['b']
        if self.transfer_fun is not None:
            self.output = self.transfer_fun(self.output)
        
        
class ResidualBlocl(NeuralNetworkUnit):
    pass


class Flatten:
    def Initialize(self, *args):
        self.output = tf.reshape(self.input, shape=[-1, int(np.prod(self.input.__dict__['_shape'][1:]))])


# The input of this layer could only be NeuronLayer.
class BatchNormalization(NeuralNetworkUnit):
    def __init__(self, dtype=tf.float64, transfer_fun=None, epsilon=0.01, **kwargs):
        super().__init__(None, None, None)
        self.dtype = dtype
        self.epsilon = epsilon
        self.kwargs = kwargs
        self.transfer_fun = transfer_fun

    def Initialize(self, input_dim,  *args):
        self.input_dim = input_dim
        self.parameters['gamma'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,
                                                                                 shape=[self.input_dim], mean=0,
                                                                                 stddev=0.1))
        self.parameters['beta'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,
                                                                                shape=[self.input_dim], mean=0,
                                                                                stddev=0.1))

        mean, var = tf.nn.moments(self.input, [0])
        self.output = tf.nn.batch_normalization(self.input, mean, var, self.parameters['beta'],
                                                self.parameters['gamma'], self.epsilon)
        try:
            self.output = self.transfer_fun(self.output)
        except TypeError:
            self.output = self.output


class AvgPooling(NeuralNetworkUnit):
    # The shape is corresponding to each dimension of the input data. 
    def __init__(self, shape, transfer_fun=None, dtype=tf.float64, **kwargs):
        super().__init__(None, None, transfer_fun)
        self.dtype = dtype
        self.shape = shape
        self.kwargs = kwargs

    def Initialize(self, *args):
        self.output = tf.nn.avg_pool(value=self.input, ksize=self.shape, strides=self.kwargs.get('strides', [1, 1, 1, 1]),
                                     padding=self.kwargs.get('padding', 'SAME'))
    
   
class MaxPooling(NeuralNetworkUnit):
    # The shape is corresponding to each dimension of the input data. 
    def __init__(self, shape, transfer_fun=None, dtype=tf.float64, **kwargs):
        super().__init__(None,None,transfer_fun)
        self.dtype = dtype
        self.shape = shape
        self.kwargs = kwargs
    def Initialize(self, *args):
        self.output = tf.nn.avg_pool(value=self.input, ksize=self.shape,
                                     strides=self.kwargs.get('strides', [1, 1, 1, 1]),
                                     padding=self.kwargs.get('padding', 'SAME'))
    
class Dropout(NeuralNetworkUnit):
    def __init__(self, keep_prob, transfer_fun=None, dtype=tf.float64, **kwargs):
        super().__init__(None,None,transfer_fun)
        self.dtype = dtype
        self.kwargs = kwargs
        self.keep_prob = keep_prob
    def Initialize(self, *args):
        self.output = tf.nn.dropout(self.input, keep_prob=self.keep_prob)



