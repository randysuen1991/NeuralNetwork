## Neural Network
This is a framework of Neural Network for myself. This framework could implement ordinary neural network and convolution neural network.
# Autoencoder example
In example.py file, I demonstrate autoencoder based on the MNIST dataset in example1. We could
see that as the number of iteration goes up, the loss would go down. Also the reconstruction
would be clearer if the number of iteration is larger. The following graph is the loss and the
MNIST original images versus the reconstructed ones.
![](https://github.com/randysuen1991/Neural-Network/blob/master/figures/autoencoder_mnist_loss.png)
![](https://github.com/randysuen1991/Neural-Network/blob/master/figures/autoencoder_mnist.png) 

# Classification example
In example.py file, I use convolution neural network to classify MNIST dataset in example2.
![](https://github.com/randysuen1991/Neural-Network/blob/master/figures/convolution_mnist_loss.png)
In this example, our classification accuracy is 0.85, in which we classify 20 testing images.