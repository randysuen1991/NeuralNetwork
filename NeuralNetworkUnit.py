import tensorflow as tf
import numpy as np


class NeuralNetworkUnit:
    def __init__(self, hidden_dim, input_dim, transfer_fun, name, dtype=tf.float64):
        # father and son store the upper layer and the lower layer of this unit.
        # if the layer is the first layer, then it is the root; if the layer is the last one, it is the leaf.
        self.name = name
        self.father = None
        self.sons = dict()
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.transfer_fun = transfer_fun
        self.parameters = dict()
        self.output = None
        self.on_train = None
        self.input = None

    def __add__(self, other):
        return Identity(self.output + other.output)

    def __sub__(self, other):
        return Identity(self.output - other.output)


class NeuronLayer(NeuralNetworkUnit):
    def __init__(self, hidden_dim, input_dim=None, transfer_fun=None, name=None, dtype=tf.float64):
        super().__init__(hidden_dim, input_dim, transfer_fun=transfer_fun, dtype=dtype, name=name)

    def Initialize(self, input_dim, counter, on_train, graph=None, **kwargs):
        self.input_dim = input_dim
        counter['Dense'] += 1
        if graph is None:
            graph = tf.get_default_graph()
        with graph.as_default():
            self.parameters['w'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,
                                                                                 shape=(self.input_dim,
                                                                                        self.hidden_dim),
                                                                                 mean=0,
                                                                                 stddev=0.1))

            self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,
                                                                                 shape=(1, self.hidden_dim),
                                                                                 mean=0,
                                                                                 stddev=0.1))
            self.output = tf.matmul(self.input, self.parameters['w']) + self.parameters['b']
            try:
                self.output = self.transfer_fun(self.output)
            except TypeError:
                self.output = self.output


class Identity(NeuralNetworkUnit):
    def __init__(self, input, hidden_dim=None, input_dim=None, transfer_fun=None, name=None, dtype=tf.float64):
        super().__init__(hidden_dim, input_dim, transfer_fun=transfer_fun, dtype=dtype, name=name)
        self.input = input

    def Initialize(self, input_dim, counter, on_train, graph=None, **kwargs):
        self.input_dim = input_dim
        if graph is None:
            graph = tf.get_default_graph()
        with graph.as_default():
            self.output = tf.identity(self.input)


class SoftMaxLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def Initialize(self, **kwargs):
        sum_exp = tf.reduce_sum(tf.exp(self.input), axis=1)
        sum_exp = tf.expand_dims(sum_exp, axis=1)
        self.output = tf.divide(tf.exp(self.input), sum_exp)


class ConvolutionUnit(NeuralNetworkUnit):
    # The shape parameter should be (height, width, num filters)
    def __init__(self, shape, transfer_fun=None, dtype=tf.float64, **kwargs):
        super().__init__(None, None, transfer_fun=transfer_fun, dtype=dtype)
        self.shape = shape
        self.kwargs = kwargs

    def Initialize(self, num_channels, **kwargs):
        shape = list(self.shape)
        shape.insert(2, num_channels)
        shape = tuple(shape)
        self.parameters['w'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,
                                                                             shape=shape, mean=0, stddev=0.1))
        self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,
                                                                             shape=(shape[-1],), mean=0, stddev=0.1))
        self.output = tf.nn.conv2d(self.input, self.parameters['w'], strides=self.kwargs.get('strides', [1, 1, 1, 1]),
                                   padding=self.kwargs.get('padding', 'SAME'))
        self.output = self.output + self.parameters['b']
        if self.transfer_fun is not None:
            self.output = self.transfer_fun(self.output)


class Reduce_Mean(NeuralNetworkUnit):
    def __init__(self, hidden_dim=None, input_dim=None, transfer_fun=None, name=None, dtype=tf.float64):
        super().__init__(hidden_dim, input_dim, transfer_fun=transfer_fun, dtype=dtype, name=name)

    def Initialize(self, input_dim=None, counter=None, on_train=None, graph=None, **kwargs):
        self.output = self.input - tf.reduce_mean(self.input, axis=1, keepdims=True)


class ResidualBlock(NeuralNetworkUnit):
    pass


class Flatten:
    def __init__(self):
        self.input = None
        self.output = None

    def Initialize(self, **kwargs):
        self.output = tf.reshape(self.input, shape=[-1, int(np.prod(self.input.__dict__['_shape'][1:]))])


# The input of this layer could only be NeuronLayer.
class BatchNormalization(NeuralNetworkUnit):
    def __init__(self, dtype=tf.float64, transfer_fun=None, epsilon=0.01, moving_decay=0.99, name=None, **kwargs):
        super().__init__(None, None, transfer_fun=transfer_fun, name=name, dtype=dtype)
        self.epsilon = epsilon
        self.kwargs = kwargs
        self.moving_decay = moving_decay

    def Initialize(self, input_dim, counter, on_train, graph=None, **kwargs):
        counter['BatchNormalization'] += 1
        self.input_dim = input_dim
        if graph is None:
            graph = tf.get_default_graph()
        with graph.as_default():
            self.output = tf.layers.batch_normalization(self.input, training=on_train)
            try:
                self.output = self.transfer_fun(self.output)
            except TypeError:
                self.output = self.output
            glb_vars = [var for var in tf.global_variables()]

        self.parameters['moving_variance'] = glb_vars[-1]
        self.parameters['moving_mean'] = glb_vars[-2]
        self.parameters['beta'] = glb_vars[-3]
        self.parameters['gamma'] = glb_vars[-4]


class AvgPooling(NeuralNetworkUnit):
    # The shape is corresponding to each dimension of the input data. 
    def __init__(self, shape, transfer_fun=None, dtype=tf.float64, **kwargs):
        super().__init__(None, None, transfer_fun=transfer_fun, dtype=dtype)
        self.shape = shape
        self.kwargs = kwargs

    def Initialize(self, **kwargs):
        self.output = tf.nn.avg_pool(value=self.input, ksize=self.shape,
                                     strides=self.kwargs.get('strides', [1, 1, 1, 1]),
                                     padding=self.kwargs.get('padding', 'SAME'))
    
   
class MaxPooling(NeuralNetworkUnit):
    # The shape is corresponding to each dimension of the input data. 
    def __init__(self, shape, transfer_fun=None, dtype=tf.float64, **kwargs):
        super().__init__(None, None, transfer_fun=transfer_fun, dtype=dtype)
        self.shape = shape
        self.kwargs = kwargs

    def Initialize(self, **kwargs):
        self.output = tf.nn.avg_pool(value=self.input, ksize=self.shape,
                                     strides=self.kwargs.get('strides', [1, 1, 1, 1]),
                                     padding=self.kwargs.get('padding', 'SAME'))


class Dropout(NeuralNetworkUnit):
    def __init__(self, keep_prob, transfer_fun=None, dtype=tf.float64, **kwargs):
        super().__init__(None, None, transfer_fun=transfer_fun, dtype=dtype)
        self.kwargs = kwargs
        self.keep_prob = keep_prob

    def Initialize(self, **kwargs):
        self.output = tf.nn.dropout(self.input, keep_prob=self.keep_prob)
