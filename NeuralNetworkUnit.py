import tensorflow as tf
import numpy as np


class NeuralNetworkUnit():
    def __init__(self,hidden_dim,input_dim,transfer_fun,dtype=tf.float64):
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.transfer_fun = transfer_fun
        self.parameters = dict()
        
class NeuronLayer(NeuralNetworkUnit):
    def __init__(self,hidden_dim,input_dim=None,transfer_fun=tf.sigmoid,dtype=tf.float64):
        super().__init__(hidden_dim,input_dim,transfer_fun,dtype=dtype)
    def Initialize(self,input_dim):
        self.input_dim = input_dim
        self.parameters['w'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.input_dim,self.hidden_dim),mean=0,stddev=0.1))
        self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(1,),mean=0,stddev=0.1))
        self.output = tf.matmul(self.input,self.parameters['w']) + self.parameters['b']
        self.output = self.transfer_fun(self.output)
        
class SoftMaxLayer():
    def Initialize(self,*args):
        sum_exp = tf.reduce_sum(tf.exp(self.input))
        self.output = tf.exp(self.input) / sum_exp
        
class ConvolutionUnit(NeuralNetworkUnit):
    # The shape parameter should be (height, width, num filters)
    def __init__(self,shape,transfer_fun,dtype=tf.float64,**kwargs):
        super().__init__(None,None,transfer_fun)
        self.dtype = dtype
        self.shape = shape
        self.kwargs = kwargs
    def Initialize(self,num_channels):
        shape = list(self.shape)
        shape.insert(2,num_channels)
        shape = tuple(shape)
        
        self.parameters['w'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=shape,mean=0,stddev=0.1))
        self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(shape[-1],),mean=0,stddev=0.1))
        self.output = tf.nn.conv2d(self.input,self.parameters['w'],strides = self.kwargs.get('strides',[1,1,1,1]),padding=self.kwargs.get('padding','SAME'))
        self.output = self.output + self.parameters['b']
        self.output = self.transfer_fun(self.output)
        
        
        
class ResidualBlocl(NeuralNetworkUnit):
    pass

class Flatten():
    def Initialize(self,*args):
        self.output = tf.reshape(self.input,shape=[-1,int(np.prod(self.input.__dict__['_shape'][1:]))])

class AvgPooling(NeuralNetworkUnit):
    def __init__(self,ksize,num_channels,**kwargs):
        self.ksize=ksize
        self.num_channels = num_channels
        self.kwargs = kwargs
    def Initialize(self,output):
        ksize = [1] + self.ksize
        ksize += [self.num_channels]
        self.output = tf.nn.avg_pool(value=self.outpt,ksize=ksize,strides=self.kwargs.get('strides',[1,1,1,1]),padding=self.kwargs.get('padding','SAME'))
    
   
class MaxPooling(NeuralNetworkUnit):
    def Initialize(self,output):
        ksize = [1] + ksize
        ksize += [self.num_channels]
        self.output = tf.nn.max_pool(value=self.outpt,ksize=ksize,strides=kwargs.get('strides',[1,1,1,1]),padding=kwargs.get('padding','SAME'))
    
class Dropout(NeuralNetworkUnit):
    def Dropout(self,keep_prob):
        self.output = tf.nn.dropout(self.output,keep_prob=keep_prob)