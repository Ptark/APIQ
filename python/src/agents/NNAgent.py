import pickle
from pathlib import Path

import numpy as np
from typing import Tuple, List

from python.src import Utility
from python.src.agents.Agent import Agent
from python.src.agents.neural_networks import NNUtility
from python.src.agents.neural_networks.NeuralNetwork import NeuralNetwork


class NNAgent(Agent):
    """Template for neural network based agents"""

    def __init__(self, activation_name: str, size: [int], training_steps: int):
        """Load appropriate parameters depending on environment and learning time"""
        super().__init__()
        self.nn = NeuralNetwork(activation_name, size)
        self.training_steps = training_steps
        self.nn.load_weights(training_steps)

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Calculates an action as a string of 0s and 1s from the neural network."""
        return self.get_activations_and_action(percept)[1]

    def get_activations_and_action(self, percept: Tuple[str, str]) -> Tuple[List, str]:
        """Feed percept into nn and returns activations and action"""
        observation = percept[0]
        action = ""
        activations = []
        if self.turn_counter == 0:
            self.idx = int(percept[0])
            observation = "1111"
            self.sign = 1 - 2 * (int(percept[1]) == 1)
            # load parameters for idx and training steps and sign
        reward = -2
        for action_idx in range(15):
            action_string = format(action_idx, 'b').zfill(4)
            nn_input = np.fromstring(observation + action_string, float)
            nn_output = self.nn.forward(nn_input)
            bit_string = NNUtility.narray_to_bitstr(nn_output[1][-1])
            action_reward = self.sign * Utility.get_reward_from_bitstring(bit_string)
            if action_reward >= reward:
                action = action_string
                reward = action_reward
                activations = nn_output
        self.turn_counter += 1
        return activations, action

    def train(self, environment_class) -> bool:
        """Train agent in environment for sign = 1 and sign = -1"""
        self.train_sign(environment_class, "0")
        self.train_sign(environment_class, "1")
        return True

    def train_sign(self, environment_class, sign_bit: str):
        """Train agent in environment training_steps times for sign = 1 and sign = -1"""
        training_steps = 10000
        for step in range(training_steps):
            self.turn_counter = 0
            environment = environment_class()
            for turn_number in environment.number_of_turns:
                activations, action = self.get_activations_and_action((str(environment.idx), sign_bit))
                percept = environment.calculate_percept(action)
                label = percept[1]
                dw_array, db_array = self.nn.backward(activations, label)
                self.nn.w_array = self.nn.w_array.add(dw_array)
                self.nn.b_array = self.nn.b_array.add(db_array)
                if turn_number > 0:
                    if np.log10(turn_number).is_integer():
                        self.save_weights(step)
                self.turn_counter += 1

    def save_weights(self, step: int):
        """Save weights to a file"""
        nn_data_dir = str(Path(".").resolve()) + "/python/resources/nn_data/"
        path = nn_data_dir + self.__class__.__name__ + "s" + str(step)
        weight_dict = {
            "weights": self.nn.w_array,
            "biases": self.nn.b_array
        }
        with open(path, "w+") as file:
            pickle.dump(weight_dict, file)

    def load_weights(self, step: int):
        """Loads weights from a file"""
        nn_data_dir = str(Path(".").resolve()) + "/python/resources/nn_data/"
        path = nn_data_dir + self.__class__.__name__ + "s" + str(step)
        with open(path, "r") as file:
            weight_dict = pickle.load(file)
        self.nn.w_array = weight_dict.get("weights")
        self.nn.b_array = weight_dict.get("biases")



