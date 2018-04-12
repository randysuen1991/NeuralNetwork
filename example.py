import NeuralNetworkModel as NNM
import NeuralNetworkUnit as NNU
import matplotlib.pyplot as plt

# In this example, I download mnist dataset from tensorflow and implement autoencoder to the dataset.
def example1():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
    
    X_train = mnist.train.images
    X_train = X_train[:500,:]
    
    model = NNM.NeuralNetworkModel()
    model.Build(NNU.NeuronLayer(hidden_dim=256))
    model.Build(NNU.NeuronLayer(hidden_dim=128))
    model.Build(NNU.NeuronLayer(hidden_dim=784))
    model.Fit(X_train,X_train,show_graph=True,num_steps=5000)
    
    
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
if __name__ == '__main__':
    example1()