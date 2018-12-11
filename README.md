# A Simple, Parametrizable Neural Network

This is a simple, fully-connected neural network capable of mini-batch gradient descent on abstract input data, 
implemented from scratch in Java.

## Performance
This network was trained and tested on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

784-100-50-10, Softmax, Cross-entropy: 98.598% training acc, 96.680% test acc
784-300-100-10, Softmax, Cross-entropy: 100.000% training acc, 97.960% test acc

Training on consumer hardware for comparable acc is generally slow (5+ minutes).

## Usage

There are two neural net types, `GeneralNeuralNet` and `SoftmaxCrossEntropyNeuralNet`. The former, as the name 
suggests, is more general as it accepts custom activation functions separately for inner and outer layers, as 
well as custom loss functions. The latter is hardcoded with Softmax outer layer activation and Cross-entropy loss, 
which has resulted in optimal accuracy in my testing.

The `NeuralNetTrainer` can be used to train any type of neural network using parametric mini-batch gradient descent, 
and there is an included example `MNISTTrainer` which is hardcoded to train on the MNIST dataset.

Input data should be passed as one-dimensional vectors to neural nets, and the NeuralNetTrainer should be used to 
automate gradient descent, although manual gradient descent, forward propagations, and other simple functions can 
be performed directly on neural nets.

When using the `MNISTTrainer` in its current configuration, training runs will be stored as entries in `logs/trainLog.csv`.

# Creators
Full implementation by Andriy Sheptunov
Most of the math by Andriy Sheptunov, the rest thanks to [Wikipedia](https://en.wikipedia.org/wiki/Backpropagation).
