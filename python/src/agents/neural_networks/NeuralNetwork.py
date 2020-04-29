import pickle
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
        std = 0.1
        self.size = size
        self.learning_rate = 1e-1
        self.activation_function = self.activation_function_switch.get(activation_name)[0]
        self.dactivation_function = self.activation_function_switch.get(activation_name)[1]
        # w for weights, b for bias
        self.weights = []
        self.biases = []
        for idx in range(len(self.size) - 1):
            self.weights.append(np.random.randn(self.size[idx + 1], self.size[idx]) * std)
            self.biases.append(np.zeros((self.size[idx + 1])))

    def forward(self, i: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
        pre_activations.append(np.zeros_like(i))
        layer_activations.append(i)
        for idx in range(len(self.size) - 1):
            pre_activations.append(np.dot(self.weights[idx], layer_activations[idx]))
            layer_activations.append(af(pre_activations[idx + 1]))
        activations = pre_activations, layer_activations
        return activations

    def backward(self, activations: Tuple[List[np.ndarray], List[np.ndarray]], label: np.ndarray) \
            -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
        dbiases = [len(self.size) - 1]
        for idx in range(len(self.weights)):
            dweights.append(np.zeros_like(self.weights[idx]))
            dbiases.append(np.zeros_like(self.biases[idx]))
        # L = 1/2 (label - o)^2  ->  dL/do = o - label

        current_derivative = layer_activations[-1] - label

        for idx in reversed(range(len(dweights))):
            print(len(pre_activations))
            print(len(dweights))
            current_derivative = current_derivative.dot(daf(pre_activations[idx + 1]).T)
            dweights[idx] = current_derivative.dot(activations[idx].T)
            dbiases[idx] = current_derivative
            current_derivative = current_derivative.dot(self.weights[idx])
        gradients = dweights, dbiases
        return gradients

    def save_weights(self, path: str):
        """Save weights to a file"""
        weight_dict = {
            "weights": self.weights,
            "biases": self.biases
        }
        with open(path, "w+") as file:
            pickle.dump(weight_dict, file)

    def load_weights(self, path: str):
        """Loads weights from a file"""
        with open(path, "r") as file:
            weight_dict = pickle.load(file)
        self.weights = weight_dict.get("weights")
        self.biases = weight_dict.get("biases")
