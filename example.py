import numpy as np
import tensorflow as tf
import NeuralNetworkModel as NNM
import NeuralNetworkUnit as NNU
import NeuralNetworkLoss as NNL

# In this example, I download mnist dataset from tensorflow and implement autoencoder to the dataset.
def example1():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
    
    X_train = mnist.train.images
    X_test = mnist.test.images
    
    model = NNM.NeuralNetworkModel()
    model.Build(NNU.NeuronLayer(hidden_dim=100))
    model.Build(NNU.NeuronLayer(hidden_dim=50))
    model.Build(NNU.NeuronLayer(hidden_dim=10))
    model.Build(NNU.NeuronLayer(hidden_dim=50))
    model.Build(NNU.NeuronLayer(hidden_dim=100))
    model.Build(NNU.NeuronLayer(hidden_dim=784))
    model.Fit(X_train,X_train)
if __name__ == '__main__':
    example1()