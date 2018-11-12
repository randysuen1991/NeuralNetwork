import tensorflow as tf
import NeuralNetworkUnit as NNU
import NeuralNetworkModel as NNM

class ModelOperation:

    # This method would return a model.
    @staticmethod
    def merge(model1, model2, name1,  name2, op, output_name):
        new_model = NNM.NeuralNetworkModel()
        input1 = model1.leaves[name1]
        input2 = model2.leaves[name2]
        if op == 'add':
            merged = input1 + input2
        elif op == 'sub':
            merged = input1 - input2

        new_model.input = merged
