from python.src.agents.neural_networks import NNUtility
import numpy as np


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
        self.size = size
        self.learning_rate = 1e-1
        self.activation_function = self.activation_function_switch.get(activation_name)[0]
        self.dactivation_function = self.activation_function_switch.get(activation_name)[1]
        self.w_array = np.empty(len(size) - 1)
        self.b_array = np.empty(len(size) - 1)
        # load learned parameters depending on environment and number of steps
        # w for weights, b for bias
        std = 0.1
        self.w_array[0] = np.random.randn(size[1], size[0]) * std
        self.w_array[-1] = np.random.randn(size[-1], size[-2]) * std
        for idx in range(len(self.w_array)):
            self.b_array[idx] = np.zeros(size[idx + 1])
            if idx == 0 or idx == len(self.w_array) - 1:
                pass
            else:
                self.w_array[idx] = np.random.randn(size[idx + 1], size[idx]) * std

    def load_weights(self):
        """Loads weights from a file"""

    def save_weights(self):
        """Save weights to a file"""

    def forward(self, i):
        """A forward pass through the network.
        params:
            i: The input to the network
        returns:
            activations: The activations of the layers so they don't have to be
                recomputed in the backwards pass. Also contains the output.
        """
        af = self.activation_function
        pre_activations = np.empty(len(self.size))
        layer_activations = np.empty(len(self.size))
        pre_activations[0] = 0
        layer_activations[0] = i
        for idx in range(len(pre_activations) - 1):
            pre_activations[idx + 1] = np.dot(self.w_array[idx], layer_activations[idx])
            layer_activations[idx + 1] = af(pre_activations[idx + 1])
        activations = pre_activations, layer_activations
        return activations

    def backward(self, activations, label):
        """Compute the gradients for training from the actual reward
        params:
            activations: the activations for each layer.
            label: Supervised learning label for computing gradients.
        returns:
            gradients: The computed gradients for parameter adjustment.
        """
        pre_activations, layer_activations = activations
        daf = self.dactivation_function
        dw_array = np.zeros_like(self.w_array)
        db_array = np.zeros_like(self.b_array)

        # L = 1/2 (label - o)^2  ->  dL/do = o - label
        current_derivative = layer_activations[-1] - label

        for idx in reversed(range(len(dw_array))):
            current_derivative = current_derivative.dot(daf(pre_activations[idx + 1]).T)
            dw_array[idx] = current_derivative.dot(activations[idx].T)
            db_array[idx] = current_derivative
            current_derivative = current_derivative.dot(self.w_array[idx])
        gradients = dw_array, db_array
        return gradients
