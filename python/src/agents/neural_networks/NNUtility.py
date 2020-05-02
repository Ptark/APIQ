from pathlib import Path

import numpy as np


def dtanh(array):
    """Derivative of tanh activation function."""
    for idx in range(len(array)):
        array[idx, 0] = 1 - array[idx, 0] * array[idx, 0]
    return array


def sigmoid(array: np.ndarray):
    """Sigmoid activation function."""
    for idx in range(len(array)):
        array[idx, 0] = 1 / (1 + np.exp(-array[idx, 0]))
    return array


def dsigmoid(array):
    """Derivative of sigmoid activation function"""
    for idx in range(len(array)):
        array[idx, 0] = array[idx, 0] * (1 - array[idx, 0])
    return array


def relu(array):
    """Relu activation function on an array"""
    for idx in range(len(array)):
        array[idx, 0] = max(0, array[idx, 0])
    return array


def drelu(array):
    """Derivative of relu activation function"""
    for idx in range(len(array)):
        array[idx, 0] = (array[idx, 0] > 0) * 1
    return array


def bitstr_to_narray(s: str) -> np.ndarray:
    """Turn a bitstring into a numpy array"""
    char_list = list(s)
    array = np.zeros((len(s), 1), dtype=np.longdouble)
    for idx in range(len(char_list)):
        array[idx, 0] = int(char_list[idx])
    return array


def narray_to_bitstr(narray: np.ndarray) -> str:
    """Turn a numpy array into a bitstring"""
    rewardstring = ""
    for o in narray:
        rewardstring += "0" if o[0] <= 0.5 else "1"
    return rewardstring


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent.parent


def get_nn_data() -> Path:
    """Returns nn_data_folder."""
    return get_project_root().joinpath('resources/agents/nn_data')


def loss(output, label):
    """Mean squared loss function"""
    dif = output - label
    return float(0.5 * np.dot(dif.T, dif))

