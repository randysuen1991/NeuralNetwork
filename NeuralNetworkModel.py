# import the Classifier module
import sys

if 'C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Classifier-and-Regressor' or 'C:\\Users\\randysuen\\Classifier-and-Regressor' not in sys.path :
    sys.path.append('C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Classifier-and-Regressor')
    sys.path.append('C:\\Users\\randysuen\\pycodes\\Classifier-and-Regressor')
    
import Classifier as C
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import NeuralNetworkUnit as NNU


class NeuralNetworkModel(C.Classifier):
    def __init__(self, dtype=tf.float64, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.target = tf.placeholder(dtype=dtype, shape=[None, None])
        self.sess = tf.Session()
        self.layers = list()
        # Presume the image_type being grayscales
        self.num_channels = kwargs.get('num_channels', 1)
        self.num_layers = 0
        self.kwargs = kwargs
        self.update = False
        self.on_train = None
        self.optimizer = None
        self.loss_fun = None
        self.batch_size = None
        self.mini_size = None
        self.input = None
        self.output = None
        self.loss = None
        self.train = None
        self.eval_model = None
        self.targ_model = None

    def __repr__(self):
        all_parameters = list()
        for layer in self.layers:
            parameters = layer.parameters
            for key, value in parameters.items():
                result = self.sess.run([value])
                all_parameters.append((key, result))
        return 'Parameters:{}'.format(all_parameters)

    # The following two functions connect all the layers.
    def _Initialize(self, output_dim, layerunit, **kwargs):
        layerunit.input = self.output
        layerunit.Initialize(output_dim, on_train=self.on_train)
        self.output = layerunit.output
        if len(layerunit.output.shape) == 4:
            return int(self.output.shape[3])
        else:
            return int(self.output.shape[1])
    
    def _Initialize_Variables(self, input_dim, **kwargs):
        unit = self.layers[0] 
        unit.input = self.input
        unit.Initialize(input_dim, on_train=self.on_train)
        self.output = unit.output
        if len(unit.output.shape) == 4:
            input_dim = int(unit.output.shape[3])
        else:
            input_dim = int(unit.output.shape[1])
        for unit in self.layers[1:]:
            if isinstance(unit, NNU.NeuralNetworkUnit):
                input_dim = self._Initialize(input_dim, unit)
            elif isinstance(unit, NeuralNetworkModel):
                unit.Compile(input_dim)

    def Build(self, layerunit):
        if isinstance(layerunit, NNU.BatchNormalization):
            self.update = True

        self.layers.append(layerunit)
        self.num_layers += 1    

    def Compile(self, X_train_shape, optimizer=None, loss_fun=None, loss_and_optimize=True, **kwargs):
        self.on_train = tf.placeholder(tf.bool)
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.batch_size = int(X_train_shape[0])
        # if the data are images, then the first layer should be some layers like convolution, pooling ....
        
        if len(X_train_shape) == 4:
            img_size = self.kwargs.get('img_size')
            # [None,None,None,None]=[batch_size,length,width,num channels]
            if self.input is None:
                self.input = tf.placeholder(dtype=self.dtype, shape=[None, img_size[0], img_size[1], X_train_shape[3]])
            self.output = self.input
            # Initialize the convolution with the num of channels.
            self._Initialize_Variables(int(X_train_shape[3]))
            
        else:
            if self.input is None:
                self.input = tf.placeholder(dtype=self.dtype, shape=[None, int(X_train_shape[1])])
            self.output = self.input
            self._Initialize_Variables(int(X_train_shape[1]))
            
        self.sess.run(tf.global_variables_initializer())
        
        if not loss_and_optimize:
            return 
        
        self.mini_size = kwargs.get('mini_size', X_train_shape[0])
        self.loss = self.loss_fun(output=self.output, target=self.target, batch_size=self.mini_size)

        # If there is anything needed to be updated, then...
        if self.update:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.train = self.optimizer.apply_gradients(grads_and_vars)
        else:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.train = self.optimizer.apply_gradients(grads_and_vars)

    def Fit(self, X_train, Y_train, loss_fun, num_epochs=5000,
            optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1), show_graph=False, **kwargs):

        self.Compile(X_train_shape=X_train.shape, optimizer=optimizer, loss_fun=loss_fun, kwargs=kwargs)
        train_losses = list()
        
        for i in range(num_epochs):
            training = list(zip(X_train, Y_train))
            random.shuffle(training)
            # If batch size = 1, then the training process is equal to stochastic gradient decent.
            # If it is equal to the number of the training set,
            # then it is equal to the batch gradient decent(classic gradient descent).
            # Otherwise, it is equal to mini-batch gradient descent.
            num_batch = X_train.shape[0] // self.mini_size
            loss_list = []
            for partition in np.array_split(training, num_batch):
                partition = list(zip(*partition))
                X_train_partition = np.array(partition[0])
                Y_train_partition = np.array(partition[1])
                _, train_loss = self.sess.run(fetches=[self.train, self.loss],
                                              feed_dict={self.input: X_train_partition,
                                                         self.target: Y_train_partition,
                                                         self.on_train: True})
                loss_list.append(train_loss)
                
            train_losses.append(np.mean(loss_list))
            if show_graph:
                # Display an update every 50 iterations
                if i % 50 == 0:
                    plt.plot(train_losses, '-b', label='Train loss')
                    plt.legend(loc=0)
                    plt.title('Loss')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.show()
                    print('Iteration: %d, train loss: %.4f' % (i, train_loss))

        return train_losses
    
    def Predict(self, X_test):
        results = self.sess.run(fetches=self.output, feed_dict={self.input: X_test,
                                                                self.on_train: False})
        return results
    
    # This is a function for evaluating the accuracy of the classifier.
    def Evaluate(self, X_test, Y_test):
        predictions = self.Predict(X_test)
        predictions = np.argmax(predictions, axis=1)
        count = 0
        correct_results = []
        for iteration, prediction in enumerate(predictions):
            if Y_test[iteration, prediction] == 1:
                count += 1
                correct_results.append(True)
            else:
                correct_results.append(False)
            
        return count/X_test.shape[0], predictions, correct_results

    def Split(self, num_channels):
        channels = []
        for i in range(num_channels):
            channel = NeuralNetworkModel()
            channel.input = self.output
            self.layers.append(channel)
            channels.append(channel)

        return channels

    def Print_Output_Detail(self, X_test):
        for layer in self.layers:
            print(layer)
            print('input:')
            layer_input, layer_output = self.sess.run([layer.input, layer.output],
                                                      feed_dict={self.input: X_test,
                                                                 self.on_train: False})
            print(layer_input)
            print('output:')
            print(layer_output)
