import numpy as np


def dtanh(x):
    """Derivative of tanh activation function."""
    return 1 - x*x


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    """Derivative of sigmoid activation function"""
    return x * (1 - x)


def relu(x):
    """Relu activation function"""
    return np.max(0, x)


def drelu(x):
    """Derivative of relu activation function"""
    return (x > 0) * 1
