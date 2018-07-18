import tensorflow as tf
import numpy as np


# The difference between NeuralNetworkOperation and NeuralNetworkUnit is the former one might have interaction with
# other models.
class NeuralNetworkOperation:
    def __init__(self):
        self.input = None
        self.output = None
        self.models = list()

    def Initialize(self):
        raise NotImplemented

class Merge(NeuralNetworkOperation):
        if op == 'add':
            return model1 + model2
        elif op == 'sub':
            return model1 - model2
        elif op == 'concat':
            new_model = NeuralNetworkModel()
            new_model.input = tf.concat([model1.output, model2.output], axis=1)
            new_model.output = new_model.input
            return new_model

    def Reduce_Mean(self):
