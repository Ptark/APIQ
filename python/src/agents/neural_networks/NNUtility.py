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


def str_to_narray(s):
    """Turn a bitstring into a numpy array neural network input"""
    return np.fromstring(s, float)


def narray_to_bitstr(narray):
    """Turn a numpy array neural network output into a bitstring"""
    rewardstring = ""
    for o in narray:
        rewardstring += "0" if o <= 0.5 else "1"
    return rewardstring
