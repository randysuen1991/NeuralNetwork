import numpy as np
import tensorflow as tf
import NeuralNetworkModel as NNM
import NeuralNetworkUnit as NNU
import NeuralNetworkLoss as NNL
def example1():
    model = NNM.NeuralNetworkModel()
    model.Build(NNU.NeuronLayer)
    model.Build(NNU.NeuronLayer)
if __name__ == '__main__':
    pass