import numpy as np
import pytest
from numpy.linalg import norm

from python.src.agents.neural_networks import NNUtility
from python.src.agents.neural_networks.NeuralNetwork import NeuralNetwork

activation_layers_inputs_label_list = [
    ("relu", [4, 2], "1111", "01")
]


@pytest.mark.parametrize("activation, layers, input_str, label_str", activation_layers_inputs_label_list)
def test_loss(activation, layers, input_str, label_str):
    """Calculates numerical biases and compares them to """
    epsilon = 1e-7
    nn = NeuralNetwork(activation, layers)
    nn_input = NNUtility.bitstr_to_narray(input_str)
    nn_output = nn.forward(nn_input)
    o = nn_output[1][-1]
    label = NNUtility.bitstr_to_narray(label_str)
    loss = NNUtility.loss(o, label)
    nn.weights[0][0][0] += epsilon
    loss_plus = NNUtility.loss(nn.forward(nn_input)[1][-1], label)
    nn.weights[0][0][0] -= 2 * epsilon
    loss_minus = NNUtility.loss(nn.forward(nn_input)[1][-1], label)
    dif = loss - (loss_plus + loss_minus) / 2
    assert dif < epsilon


@pytest.mark.parametrize("activation, layers, input_str, label_str", activation_layers_inputs_label_list)
def test_gradient(activation, layers, input_str, label_str):
    """Calculates numerical biases and compares them to """
    epsilon = 1e-7
    nn = NeuralNetwork(activation, layers)
    nn_input = NNUtility.bitstr_to_narray(input_str)
    nn_output = nn.forward(nn_input)
    label = NNUtility.bitstr_to_narray(label_str)
    dweights, dbiases = nn.backward(nn_output, label)
    dweight = dweights[-1][0][0]

    nn.weights[-1][0][0] += epsilon
    loss_plus = NNUtility.loss(nn.forward(nn_input)[1][-1], label)

    nn.weights[-1][0][0] -= 2 * epsilon
    loss_minus = NNUtility.loss(nn.forward(nn_input)[1][-1], label)

    dweight_approx = (loss_plus - loss_minus) / (2 * epsilon)

    nominator = norm(dweight - dweight_approx)
    denominator = norm(dweight) + norm(dweight_approx)
    dif = nominator / denominator

    assert dif < epsilon

