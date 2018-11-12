from ClassifierAndRegressor.Core import Classifier as C
from NeuralNetwork import NeuralNetworkUnit as NNU
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random


class NeuralNetworkTree:
    def __init__(self):
        self.root = None
        self.leaves = dict()
        self.height = 0


class NeuralNetworkModel(C.Classifier):
    def __init__(self, dtype=tf.float64, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.graph = kwargs.get('graph', tf.Graph())
        self.sess = tf.Session(graph=self.graph)
        self.NNTree = NeuralNetworkTree()
        # Presume the image_type being grayscales
        self.num_channels = kwargs.get('num_channels', 1)
        self.num_layers = 0
        self.loss_and_optimize = None
        self.kwargs = kwargs
        self.update = False
        self.optimizer = None
        self.target = None
        self.loss_fun = None
        self.batch_size = None
        self.mini_batch = None
        self.output = None
        self._input = None
        self.loss = None
        self.train = None
        self.on_train = None
        self.counter = {'Dense': 0, 'BatchNormalization': 0, 'Convolution': 0, 'MaxPooling': 0, 'AvgPooling': 0,
                        'Dropout': 0, 'Flatten': 0, 'Identity': 0, 'SoftMax': 0}

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value
        self.output = self._input

    def __repr__(self):
        all_parameters = list()
        for layer in self.layers:
            parameters = layer.parameters
            for key, value in parameters.items():
                result = self.sess.run([value])
                all_parameters.append((key, result))
        return 'Parameters:{}'.format(all_parameters)

    def __sub__(self, model):
        new_model = NeuralNetworkModel()
        new_model.input = self.output - model.output
        new_model.output = new_model.input
        return new_model

    def __add__(self, model):
        new_model = NeuralNetworkModel()
        new_model.input = self.output + model.output
        new_model.output = new_model.input
        return new_model

    def build(self, layer, name='last', **kwargs):
        # layer is going to connect to the layer with the name.
        if layer.name is None:
            layer.name = name
        if isinstance(layer, NNU.BatchNormalization):
            self.update = True

        if self.NNTree.height == 0:
            input_dim = kwargs.get('input_dim', None)
            if input_dim is None:
                raise ValueError('You should specify the input dimension of the first layer.')
            self.NNTree.root = layer
            self.NNTree.leaves[layer.name] = layer
            self.NNTree.height += 1
            shape = [None]
            if type(input_dim) is tuple:
                input_dim = list(input_dim)
            elif type(input_dim) is int or type(input_dim) is float:
                input_dim = [input_dim]
            shape += input_dim
            input_dim = shape
            with self.graph.as_default():
                self.on_train = tf.placeholder(tf.bool)
                self.input = tf.placeholder(dtype=self.dtype, shape=shape)
            layer.input = self.input
        else:
            self.NNTree.height += 1
            father = self.NNTree.leaves[name]
            father.sons[layer.name] = layer
            layer.father = father
            if kwargs.get('pop', True):
                self.NNTree.leaves.pop(name)
            self.NNTree.leaves[layer.name] = layer
            layer.input = father.output
            input_dim = father.output.shape

        layer.initialize(input_dim=input_dim, counter=self.counter, on_train=self.on_train, graph=self.graph)

        # If there are multiple outputs, the output attribute would be a dictionary. Otherwise, it would be a Tensor.
        outputs = dict()
        for key, value in self.NNTree.leaves.items():
            outputs[key] = value
        if len(outputs) == 1:
            self.output = self.NNTree.leaves[layer.name].output
        else:
            self.output = outputs

    # If the loss fun takes more than two values to compute the loss, those extra  should be stored in kwargs.
    def compile(self, optimizer=None, loss_fun=None, loss_and_optimize=True, **kwargs):
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        self.loss_and_optimize = loss_and_optimize
        if not self.loss_and_optimize:
            return
        with self.graph.as_default():
            self.target = tf.placeholder(dtype=self.dtype, shape=[None, None])
            self.loss = self.loss_fun(output=self.output, target=self.target, batch_size=self.mini_batch,
                                      dtype=self.dtype, **kwargs)

            # If there is anything needed to be updated, then...
            if self.update:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grads_and_vars = self.optimizer.compute_gradients(self.loss)
                    self.train = self.optimizer.apply_gradients(grads_and_vars)
            else:
                grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.train = self.optimizer.apply_gradients(grads_and_vars)

    def fit(self, x_train, y_train, loss_fun, num_epochs=5000,
            optimizer=tf.train.GradientDescentOptimizer, learning_rate=0.001, show_graph=False, **kwargs):
        optimizer = optimizer(learning_rate=learning_rate)
        self.mini_batch = kwargs.get('mini_batch', int(x_train.shape[0]))
        self.compile(optimizer=optimizer, loss_fun=loss_fun)
        train_losses = list()
        train_loss = None
        epoch_list = list()
        for i in range(num_epochs):
            training = list(zip(x_train, y_train))
            random.shuffle(training)
            # If batch size = 1, then the training process is equal to stochastic gradient decent.
            # If it is equal to the number of the training set,
            # then it is equal to the batch gradient decent(classic gradient descent).
            # Otherwise, it is equal to mini-batch gradient descent.
            num_batch = x_train.shape[0] // self.mini_batch
            loss_list = []
            for partition in np.array_split(training, num_batch):
                partition = list(zip(*partition))
                x_train_partition = np.array(partition[0])
                y_train_partition = np.array(partition[1])
                _, train_loss = self.sess.run(fetches=[self.train, self.loss],
                                              feed_dict={self.input: x_train_partition,
                                                         self.target: y_train_partition,
                                                         self.on_train: True})
                epoch_list.append(train_loss[0])
                loss_list.append(train_loss[0])

            train_losses.append(np.mean(loss_list))
            print('Loss of epoch {} is {}'.format(i, np.mean(epoch_list)))
            epoch_list = list()
            if show_graph:
                # Display an update every 50 iterations
                if i % 10 == 0 and i != 0:
                    print(len(train_losses))
                    plt.plot(train_losses, '-b', label='Train loss')
                    plt.legend(loc=0)
                    plt.title('Loss')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.show()
                    print('Iteration: %d, train loss: %.4f' % (i, train_loss))

        return train_losses
    
    def predict(self, x_test):
        results = self.sess.run(fetches=self.output, feed_dict={self.input: x_test,
                                                                self.on_train: False})
        return results
    
    # This is a function for evaluating the accuracy of the classifier.
    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)
        predictions = np.argmax(predictions, axis=1)
        count = 0
        correct_results = []
        for iteration, prediction in enumerate(predictions):
            if y_test[iteration, prediction] == 1:
                count += 1
                correct_results.append(True)
            else:
                correct_results.append(False)
            
        return count/x_test.shape[0], predictions, correct_results

    def split(self, names, name='last', **kwargs):
        target = self.NNTree.leaves[name]
        self.NNTree.leaves.pop(name)
        for n in names:
            split = NNU.Identity(input=target.output)
            split.initialize(input_dim=target.input_dim, counter=self.counter, on_train=self.on_train, graph=self.graph)
            split.name = n
            self.NNTree.leaves[n] = split
            try:
                target.sons[n] = split
            except AttributeError:
                pass

        outputs = dict()
        for key, value in self.NNTree.leaves.items():
            outputs[key] = value
        self.output = outputs

    # merge the splits, in a model, into one.
    def merge(self, op, names, output_name='merged_last'):
        if op == 'add':
            output = self.NNTree.leaves[names[0]]
            merged = NNU.Identity(output.output)
            merged.initialize(input_dim=None, counter=self.counter, on_train=self.on_train)

            for n in names[1:]:
                output = self.NNTree.leaves[n]
                merged += output

            for n in names:
                output = self.NNTree.leaves[n]
                output.sons[n] = merged
                self.NNTree.leaves.pop(n)

        elif op == 'concat':
            pass
        if output_name in self.NNTree.leaves:
            raise ValueError('You should specify a new name to the merged output.')

        self.NNTree.leaves[output_name] = merged

        outputs = dict()
        for key, value in self.NNTree.leaves.items():
            outputs[key] = value
        if len(outputs) == 1:
            self.output = self.NNTree.leaves[output_name].output
        else:
            self.output = outputs

    def print_output_detail(self, x_test, **kwargs):
        layer = self.NNTree.root
        self._print_output_detail_recursive(layer, x_test, sess=kwargs.get('sess', None))

    def _print_output_detail_recursive(self, layer, x_test, sess):
        if sess is not None:
            layer_input, layer_output = sess.run([layer.input, layer.output],
                                                 feed_dict={self.input: x_test,
                                                            self.on_train: False})
        else:
            layer_input, layer_output = self.sess.run([layer.input, layer.output],
                                                      feed_dict={self.input: x_test,
                                                                 self.on_train: False})
        print(layer)
        print('input:', layer_input)
        print('output:', layer_output)
        print('sons:', layer.sons)
        for _, son in layer.sons.items():
            self._print_output_detail_recursive(son, x_test, sess)

    def print_parameters(self, **kwargs):
        layer = self.NNTree.root
        self._print_parameters_recursive(layer, sess=kwargs.get('sess', None))

    def _print_parameters_recursive(self, layer, sess):
        if sess is not None:
            for key, parameter in layer.parameters.items():
                print(parameter.name)
                value = sess.run(parameter)
                print(value)
        else:
            for key, parameter in layer.parameters.items():
                print(parameter.name)
                value = self.sess.run(parameter)
                print(value)

        for _, son in layer.sons.items():
            self._print_parameters_recursive(son, sess)
