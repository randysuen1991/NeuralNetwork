import NeuralNetwork.NeuralNetworkModel as NNM
import NeuralNetwork.NeuralNetworkUnit as NNU
import NeuralNetwork.NeuralNetworkLoss as NNL
import DimensionReductionApproaches.UtilFun as UF
import tensorflow as tf

import numpy as np
import os

# In this example, I download mnist dataset from tensorflow and implement autoencoder to the dataset.
def example1():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x_train = mnist.train.images
    # To reduce computational cost, we let the training size be 500.
    x_train = x_train[:500, :]
    
    model = NNM.NeuralNetworkModel()
    model.build(NNU.NeuronLayer(hidden_dim=10), input_dim=784)
    model.build(NNU.BatchNormalization())
    model.build(NNU.NeuronLayer(hidden_dim=5, transfer_fun=tf.nn.sigmoid))
    model.build(NNU.BatchNormalization())
    model.build(NNU.NeuronLayer(hidden_dim=784))

    import time
    t1 = time.time()
    model.Fit(x_train, x_train, show_graph=False, num_epochs=500,
              mini_size=40, loss_fun=NNL.NeuralNetworkLoss.meansquared)
    print(time.time()-t1)
    
    n = 1  # how many digits we will display
    x_test = mnist.test.images
    x_test = x_test[:n, :]

    results = model.predict(x_test)
    model.print_output_detail(x_test)
    # model.Print_Parameters()
    # plot the testing images.
    
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
        # display original
        # ax = plt.subplot(2, n, i + 1)
        # plt.imshow(x_test[i].reshape(28, 28))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        # ax = plt.subplot(2, n, i + 1 + n)
        # plt.imshow(results[i].reshape(28, 28))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

    # plt.show()
    # print(model)


# In this example, I use Convolution Neural Network to classify the MNIST images.
def example2():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x_train = mnist.train.images
    # To reduce computational cost, we let the training size be 500.
    x_train = x_train[:500, :]
    y_train = mnist.train.labels
    y_train = y_train[:500, :]
    x_train = UF.vectors2imgs(x_train, (None, 28, 28, 1))
    model = NNM.NeuralNetworkModel(dtype=tf.float32, img_size=(28, 28))
    # shape=(5,5,3) means the kernel's height=5 width=5 num of ker=3
    model.build(NNU.ConvolutionUnit(dtype=tf.float32, shape=(5, 5, 3), transfer_fun=tf.tanh))
    model.build(NNU.AvgPooling(dtype=tf.float32, shape=(1, 4, 4, 1)))
    model.build(NNU.Dropout(keep_prob=0.5))
    model.build(NNU.Flatten())
    model.build(NNU.NeuronLayer(hidden_dim=10, dtype=tf.float32))
    model.build(NNU.SoftMaxLayer())
    model.fit(x_train, y_train, loss_fun=NNL.NeuralNetworkLoss.crossentropy, show_graph=True, num_epochs=1000)

    x_test = mnist.test.images
    y_test = mnist.test.labels
    x_test = x_test[0:20, :]
    y_test = y_test[0:20, :]
    x_test = UF.vectors2imgs(x_test, (None, 28, 28, 1))
    print(model.Evaluate(x_test, y_test))


def example3():
    def load_data(filename):
        this_dir, _ = os.path.split(__file__)
        data_path = os.path.join(this_dir, 'data', filename)
        return np.load(data_path)

    imgs, labels, shape = load_data('FERET.npy')
    x_train, y_train, x_test, y_test = UF.split_train_test(imgs, labels, 2)
    y_train = UF.OneHot(y_train)
    y_test = UF.OneHot(y_test)
    model = NNM.NeuralNetworkModel(dtype=tf.float64)
    # shape=(5,5,3) means the kernel's height=5 width=5 num of ker=3.
    # first layer should give the input dim, which in this case is (image height, image width, num channel).
    model.build(NNU.ConvolutionUnit(dtype=tf.float64, shape=(4, 4, 2),
                                    transfer_fun=tf.tanh), input_dim=x_train.shape[1:])
    # model.build(NNU.BatchNormalization())
    # model.build(NNU.Dropout(keep_prob=0.8))
    # model.build(NNU.ConvolutionUnit(dtype=tf.float64, shape=(2, 2, 4),
    #                                 transfer_fun=tf.sigmoid))

    # model.build(NNU.AvgPooling(dtype=tf.float64, shape=(1, 4, 4, 1)))
    model.build(NNU.Dropout(keep_prob=0.8))
    model.build(NNU.Flatten())
    model.build(NNU.NeuronLayer(hidden_dim=shape[1], dtype=tf.float64))
    model.build(NNU.BatchNormalization())
    model.build(NNU.SoftMaxLayer())
    model.fit(x_train, y_train, loss_fun=NNL.NeuralNetworkLoss.crossentropy, show_graph=False, num_epochs=501,
              mini_batch=10, learning_rate=0.5)
    acc, result, correctness = model.evaluate(x_test, y_test)
    print(acc, result, correctness)
#    print(model.sess.run(model.layers[0].parameters['w'][:,:,0,0]))
#    print(model.layers[0].parameters['w'].shape)


def example4():
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    input = tf.constant([[1.0, 2.0, 3.0], [4.0, 7.0, 5.0]])
    batch_mean, batch_var = tf.nn.moments(input, axes=[0])
    print(sess.run(batch_mean), sess.run(batch_var), '\n')
    output = tf.nn.batch_normalization(input, batch_mean, batch_var, 3, 2, 0)
    print(sess.run(output))

def example5():
    pass

if __name__ == '__main__':
    example3()
