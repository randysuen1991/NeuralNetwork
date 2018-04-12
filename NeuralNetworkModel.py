# import the Classifier module
import sys
if 'C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Classifier-and-Regressor' not in sys.path :
    sys.path.append('C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Classifier-and-Regressor')
import Classifier as C
import NeuralNetworkLoss as NNL
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetworkModel(C.Classifier):
    
        
    def __init__(self,**kwargs):
        super().__init__()
        #[None,None,None,None]=[batch_size,length,width,rgb order]
        self.input = tf.placeholder(dtype=tf.float32,shape=[None,None,None,None])
        self.output = self.input
        #[None,None] = [batch_size,label]
        self.target = tf.placeholder(dtype=tf.float32,shape=[None,None])
        self.sess = tf.Session()
        self.parameters = dict()
        self.layers = list()
        # Presume the image_type being grayscales
        self.num_channels = kwargs.get('num_channels',1)
        self.num_layers = 0
        
        
        
        
    # The following two functions connect all the layers.    
    def _Initialize(self,output_dim,recurrentunit):
        recurrentunit.input = self.output
        recurrentunit.Initialize(output_dim)
        self.output = recurrentunit.output
        return int(self.output.shape[1])
    
    def _Initialize_Variables(self,input_dim):
        unit = self.layers[0]
        unit.input = self.input
        if input_dim != None:
            unit.Initialize(input_dim)
        else :
            unit.Initialize()
        self.output = unit.output
        input_dim = int(unit.output.shape[2])
        for unit in self.layers[1:] :
            input_dim = self._Initialize(input_dim,unit)
        
        
    def Build(self,layerunit):
        self.layers.append(layerunit)
        self.num_layers += 1    
        
    def Fit(self,X_train,Y_train,num_steps=5000,loss_fun=NNL.NeuralNetworkLoss.MeanSqaured,
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),show_graph=False,**kwargs):
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        
        self.batch_size = int(X_train.shape[0])
        # if the data are images, then the first layer should be some layers like convolution, pooling ....
        if len(X_train.shape) == 4 :
            self._Initialize_Variables(None)
        else:
            self._Initialize_Variables(int(X_train.shape[1]))
        
        
        loss = self.loss_fun(output=self.output,target=self.target,batch_size=self.batch_size)
        self.sess.run(tf.global_variables_initializer())
        grads_and_vars = self.optimizer.compute_gradients(loss)
        train = self.optimizer.apply_gradients(grads_and_vars)
        train_losses = list()
        for i in range(num_steps):
            _, train_loss = self.sess.run(fetches=[train,loss],feed_dict={self.input:X_train,self.target:Y_train})
            train_losses.append(train_loss)

            if show_graph :
#           Display an update every 50 iterations
                if i % 50 == 0:
                    plt.plot(train_losses, '-b', label='Train loss')
                    plt.legend(loc=0)
                    plt.title('Loss')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.show()
                    print('Iteration: %d, train loss: %.4f' % (i, train_loss))
        return train_losses