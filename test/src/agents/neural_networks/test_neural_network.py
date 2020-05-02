import numpy as np
import pytest

from python.src.agents.neural_networks import NNUtility
from python.src.agents.neural_networks.NeuralNetwork import NeuralNetwork


def test_gradients():
    """Calculates numerical biases and compares them to """
    nn = NeuralNetwork("sigmoid", [4, 3])
    nn_input = NNUtility.bitstr_to_narray("1101")
    nn_output = nn.forward(nn_input)
    label = NNUtility.bitstr_to_narray("101")
    dweights, dbiases = nn.backward(nn_output, label)
    delta = 0.0000001
    rel_tol = 0.04
    for idx in range(len(nn.weights)):
        for i in range(len(nn.weights[idx])):
            # Calculate numerical biases and compare to backprop
            original_bias = nn.biases[idx][i]
            nn.biases[idx][i] += delta
            loss_plus = NNUtility.loss(nn.forward(nn_input)[1][-1], label)
            nn.biases[idx][i] -= 2 * delta
            loss_minus = NNUtility.loss(nn.forward(nn_input)[1][-1], label)
            dbias_numerical = (loss_plus - loss_minus) / (2 * delta)
            assert np.isclose(dbias_numerical, dbiases[idx][i], rtol=rel_tol)
            nn.biases[idx][i] = original_bias
            for j in range(len(nn.weights[idx][i])):
                # Calculate numerical weights compare to backprop
                original_weight = nn.weights[idx][i][j]
                nn.weights[idx][i][j] += delta
                loss_plus = NNUtility.loss(nn.forward(nn_input)[1][-1], label)
                nn.weights[idx][i][j] -= 2 * delta
                loss_minus = NNUtility.loss(nn.forward(nn_input)[1][-1], label)
                dweight_numerical = (loss_plus - loss_minus) / (2 * delta)
                assert np.isclose(dweight_numerical, dweights[idx][i][j], rtol=rel_tol)
                nn.weights[idx][i][j] = original_weight

