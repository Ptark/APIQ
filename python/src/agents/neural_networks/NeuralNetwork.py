import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from python.src.agents.neural_networks import NNUtility


class NeuralNetwork:

    activation_function_switch = {
        "sigmoid": [NNUtility.sigmoid, NNUtility.dsigmoid],
        "relu": [NNUtility.relu, NNUtility.drelu],
        "tanh": [np.tanh, NNUtility.dtanh]
    }

    def __init__(self, activation_name: str, size: [int]):
        """Initiate basic neural network with parameters
        param:
            activation_name: name for switch which chooses activation function
            hidden_size: size of the hidden layer [number of layers, neurons per layer]
        """
        std = 0.01
        self.size = size
        self.learning_rate = 1e-1
        self.activation_function = self.activation_function_switch.get(activation_name)[0]
        self.dactivation_function = self.activation_function_switch.get(activation_name)[1]
        # w for weights, b for bias
        self.weights = []
        self.biases = []
        for idx in range(len(self.size) - 1):
            self.weights.append(np.random.randn(self.size[idx + 1], self.size[idx]) * std)
            self.weights[idx].astype(np.longdouble)
            self.biases.append(np.zeros((self.size[idx + 1], 1), dtype=np.longdouble))

    def forward(self, nn_input: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """A forward pass through the network.
        params:
            i: The input to the network
        returns:
            activations: The activations of the layers so they don't have to be
                recomputed in the backwards pass. Also contains the output.
        """
        af = self.activation_function
        pre_activations = []
        layer_activations = []
        pre_activations.append(np.zeros_like(nn_input, dtype=np.longdouble))
        layer_activations.append(nn_input)
        layer_activations[0].astype(np.longdouble)
        for idx in range(len(self.size) - 1):
            pre_activations.append(np.dot(self.weights[idx], layer_activations[idx]) + self.biases[idx])
            layer_activations.append(af(pre_activations[idx + 1]))
        activations = pre_activations, layer_activations
        return activations

    def backward(self, activations: Tuple[List[np.ndarray], List[np.ndarray]], label: np.ndarray):
        """Compute the gradients for training from the actual reward
        params:
            activations: the activations for each layer.
            label: Supervised learning label for computing gradients.
        returns:
            gradients: The computed gradients for parameter adjustment.
        """
        pre_activations, layer_activations = activations
        daf = self.dactivation_function
        dweights = []
        dbiases = []
        for idx in range(len(self.weights)):
            dweights.append(np.zeros_like(self.weights[idx], dtype=np.longdouble))
            dbiases.append(np.zeros_like(self.biases[idx], dtype=np.longdouble))
        # L = 1/2 (label - o)^2  ->  dL/do = o - label
        # c: current_derivative, d: dactivation_function_used

        c = layer_activations[-1] - label

        for idx in reversed(range(len(dweights))):
            d = daf(pre_activations[idx + 1])
            c = c * d
            dweights[idx] = np.dot(c, layer_activations[idx].T)
            dbiases[idx] = c
            c = np.dot(self.weights[idx].T, c)
        for idx in range(len(self.weights)):
            self.weights[idx] = self.weights[idx] - dweights[idx] * self.learning_rate
            self.biases[idx] = self.biases[idx] - dbiases[idx] * self.learning_rate
        return dweights, dbiases

    def save_weights(self, path: Path):
        """Save weights to a file"""
        weight_dict = {
            "weights": self.weights,
            "biases": self.biases
        }
        pickle.dump(weight_dict, path.open("wb"))

    def load_weights(self, path: Path):
        """Loads weights from a file"""
        weight_dict = pickle.load(path.open("rb"))
        self.weights = weight_dict.get("weights")
        self.biases = weight_dict.get("biases")
