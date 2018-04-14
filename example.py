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
    mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
    
    X_train = mnist.train.images
    # To reduce computational cost, we let the training size be 500.
    X_train = X_train[:500,:]
    
    model = NNM.NeuralNetworkModel()
    model.Build(NNU.NeuronLayer(hidden_dim=256))
    model.Build(NNU.NeuronLayer(hidden_dim=128))
    model.Build(NNU.NeuronLayer(hidden_dim=784))
    model.Fit(X_train,X_train,show_graph=True,num_steps=5000,loss_fun=NNL.NeuralNetworkLoss.MeanSqaured)
    
    
    n = 10  # how many digits we will display
    X_test = mnist.test.images
    X_test = X_test[:n,:]
    
    results = model.Predict(X_test)
    
    # plot the testing images.
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
    #     display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    #     display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(results[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    
# In this example, I use Convolution Neural Network to classify the MNIST images.
def example2():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
    X_train = mnist.train.images
    # To reduce computational cost, we let the training size be 500.
    X_train = X_train[:500,:]
    Y_train = mnist.train.labels
    Y_train = Y_train[:500,:]
    X_train = UF.vectors2imgs(X_train,(None,28,28,1))
    
    model = NNM.NeuralNetworkModel(dtype=tf.float32)
    model.Build(NNU.ConvolutionUnit(dtype=tf.float32,shape=(5,5,3),transfer_fun=tf.sigmoid))
    model.Build(NNU.Flatten())
    model.Build(NNU.SoftMaxLayer())
    model.Fit(X_train,Y_train,loss_fun=NNL.NeuralNetworkLoss.CrossEntropy)

    
if __name__ == '__main__':
    example2()