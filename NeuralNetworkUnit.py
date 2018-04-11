import tensorflow as tf

class NeuralNetworkUnit():
    def __init__(self,hidden_dim,input_dim,dtype=tf.float64):
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.parameters = dict()
        
class NeuronLayer(NeuralNetworkUnit):
    def __init__(self,hidden_dim,input_dim=None,dtype=tf.float64):
        super().__init__(hidden_dim,input_dim)
    def Initialize(self,input_dim):
        self.input_dim = input_dim
        self.parameters['w'] = tf.Variable(initial_value=tf.truncated_normal(shape=(self.input_dim,1),dtype=tf.float64,mean=0,stddev=0.1))
        self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(shape=(1,),dtype=tf.float64,mean=0,stddev=0.1))
        self.output = tf.map_fn(fn=lambda output: tf.matmul(output,self.parameters['w'])+self.parameters['b'],elems = self.input)

class ConvolutionUnit():
    def __init__(self,shape,dtype=tf.float64):
        super().__init__(None,None)
        self.dtype = dtype
        self.shape = shape
    def Initialize(self,input_shape):
        self.parameters['w'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=self.shape,mean=0,stddev=0.1))
        self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=self.shape[-1],mean=0,stddev=0.1))
                
        
        