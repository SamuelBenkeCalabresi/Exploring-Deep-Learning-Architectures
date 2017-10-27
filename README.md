# Exploring Deep Learning Architectures

Implementation with pure Java of Deep Learning algorithms, graphical models and generative models such as Deep Belief Nets (DBNs), Deep feedforward neural nets (Deep MPLs), softmax regression and restricted Boltzmann machines (RBMs). 

## Abstract

Deep learning of deep neural architectures made of multiple hidden layers of non linear functions allow the learning of high-level abstractions. Deep belief nets as proposed by Hinton et al. learn through hierarchical greedy training of layers of feature detectors. The layers in this project are stochastic neural nets called restricted Boltzmann machines that would apply to each level of depth an autoencoder function for feature extractions. The model learns via an unsupervised layer-wise pre-training algorithm followed by a supervised backpropagation fine-tuning algorithm. To perform discriminative tasks, the DBN in the fine-tuning is seen as a deep feedforward neural net with a backpropagation learning algorithm to find, with a stochastic gradient descent, the global optima as a local search started by initial sensible gradients of the pre-training. The project explores and illustrates all these components and then implements them with a low-level programming language (java). Among the explorations it is shown that at each new layer of feature detector there is an improvement in the learning. Given a discriminative task, a softmax regression is exploited to maximise the probabilities of a class. It has also been discovered how the learning slow down exponentially at each new hidden layer added to the network.

## Prerequisites

MNIST database and the Optical Recognition of Handwritten Digits dataset from the repository of the University of California at Irvine (UCI dataset). They are contained in the project, but can be found as follows:
[MNIST](http://yann.lecun.com/exdb/mnist/) and
[UCI](http://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits).

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details
