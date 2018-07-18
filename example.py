import sys
if 'C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches' not in sys.path:
    sys.path.append('C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches')
import NeuralNetworkModel as NNM
import NeuralNetworkUnit as NNU
import NeuralNetworkLoss as NNL
import matplotlib.pyplot as plt
import UtilFun as UF
import tensorflow as tf
import matplotlib.pyplot as plt


# In this example, I download mnist dataset from tensorflow and implement autoencoder to the dataset.
def example1():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    
    X_train = mnist.train.images
    # To reduce computational cost, we let the training size be 500.
    X_train = X_train[:500, :]
    
    model = NNM.NeuralNetworkModel()
    model.Build(NNU.NeuronLayer(hidden_dim=10))
    model.Build(NNU.BatchNormalization())
    model.Build(NNU.NeuronLayer(hidden_dim=5, transfer_fun=tf.nn.sigmoid))
    # model.Build(NNU.BatchNormalization())
    model.Build(NNU.NeuronLayer(hidden_dim=784))
    import time
    t1 = time.time()
    model.Fit(X_train, X_train, show_graph=True, num_epochs=500,
              mini_size=40, loss_fun=NNL.NeuralNetworkLoss.MeanSqaured)
    print(time.time()-t1)
    
    n = 1  # how many digits we will display
    X_test = mnist.test.images
    X_test = X_test[:n, :]

    results = model.Predict(X_test)
    # model.Print_Output_Detail(X_test)
    # plot the testing images.
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        # plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(results[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # plt.show()
    # print(model)


# In this example, I use Convolution Neural Network to classify the MNIST images.
def example2():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    X_train = mnist.train.images
    # To reduce computational cost, we let the training size be 500.
    X_train = X_train[:500, :]
    Y_train = mnist.train.labels
    Y_train = Y_train[:500, :]
    X_train = UF.vectors2imgs(X_train, (None, 28, 28, 1))
    model = NNM.NeuralNetworkModel(dtype=tf.float32, img_size=(28, 28))
    # shape=(5,5,3) means the kernel's height=5 width=5 num of ker=3
    model.Build(NNU.ConvolutionUnit(dtype=tf.float32, shape=(5, 5, 3), transfer_fun=tf.tanh))
    model.Build(NNU.AvgPooling(dtype=tf.float32, shape=(1, 4, 4, 1)))
    model.Build(NNU.Dropout(keep_prob=0.5))
    model.Build(NNU.Flatten())
    model.Build(NNU.NeuronLayer(hidden_dim=10, dtype=tf.float32))
    model.Build(NNU.SoftMaxLayer())
    model.Fit(X_train, Y_train, loss_fun=NNL.NeuralNetworkLoss.CrossEntropy, show_graph=True, num_epochs=1000)

    X_test = mnist.test.images
    Y_test = mnist.test.labels
    X_test = X_test[0:20, :]
    Y_test = Y_test[0:20, :]
    X_test = UF.vectors2imgs(X_test, (None, 28, 28, 1))
    print(model.Evaluate(X_test, Y_test))

def example3():
    import numpy as np
    imgs, labels, shape = np.load('ORL.npy')
    X_train, Y_train, X_test, Y_test = UF.split_train_test(imgs, labels, 2)
    model = NNM.NeuralNetworkModel(dtype=tf.float32, img_size=(112, 92))
    #shape=(5,5,3) means the kernel's height=5 width=5 num of ker=3
    model.Build(NNU.ConvolutionUnit(dtype=tf.float32, shape=(5, 5, 3), transfer_fun=tf.tanh))
    model.Build(NNU.AvgPooling(dtype=tf.float32, shape=(1, 4, 4, 1)))
    model.Build(NNU.Dropout(keep_prob=0.5))
    model.Build(NNU.Flatten())
    model.Build(NNU.NeuronLayer(hidden_dim=10, dtype=tf.float32))
    model.Build(NNU.SoftMaxLayer())
    model.Fit(X_train, Y_train, loss_fun=NNL.NeuralNetworkLoss.CrossEntropy, show_graph=True, num_epochs=500)
    print(model.Evaluate(X_test, Y_test))
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


if __name__ == '__main__':
    example1()
