import numpy as np
from typing import Tuple, List, Type

from python.src import Utility
from python.src.agents.Agent import Agent
from python.src.agents.neural_networks import NNUtility
from python.src.agents.neural_networks.NeuralNetwork import NeuralNetwork
from python.src.environments.Environment import Environment


class NNAgent(Agent):
    """Template for neural network based agents"""

    training_steps = 101
    nn_data_dir = NNUtility.get_nn_data()

    def __init__(self, training_step: int, activation_name: str, size: [int]):
        """Load appropriate parameters depending on environment and learning time"""
        super().__init__(training_step)
        self.nn = NeuralNetwork(activation_name, size)

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Calculates an action as a string of 0s and 1s from the neural network."""
        return self.calculate_activations_and_action(percept)[1]

    def calculate_activations_and_action(self, percept: Tuple[str, str]) \
            -> Tuple[Tuple[List[np.ndarray], List[np.ndarray]], str]:
        """Feed percept into nn and returns activations and action"""
        observation = percept[0]
        action = ""
        activations = []
        if self.turn_counter == 0:
            observation = "1111"
        if not self.is_initialized:
            self.idx = int(percept[0])
            self.sign = 1 - 2 * (int(percept[1]) == 1)
            # load parameters for idx and training steps and sign
            if self.training_step > 0:
                filename = self.__class__.__name__ + "_" + str(self.idx) + "_" + str(self.sign) + "_" + \
                           str(self.training_step) + ".td"
                path = NNAgent.nn_data_dir.joinpath(filename)
                self.nn.load_weights(path)
            self.is_initialized = True
        reward = -2
        for action_idx in range(15):
            action_string = format(action_idx, 'b').zfill(4)
            nn_input = NNUtility.bitstr_to_narray(observation + action_string)
            nn_output = self.nn.forward(nn_input)
            bit_string = NNUtility.narray_to_bitstr(nn_output[1][-1])
            action_reward = self.sign * Utility.get_reward_from_bitstring(bit_string)
            if action_reward >= reward:
                action = action_string
                reward = action_reward
                activations = nn_output
        self.turn_counter += 1
        return activations, action

    def train(self, environment_class: Type[Environment], sign_bit: str):
        """Train agent in environment training_steps times"""
        for step in range(NNAgent.training_steps):
            self.turn_counter = 0
            environment = environment_class()
            for turn_number in range(environment.number_of_turns):
                activations, action = self.calculate_activations_and_action((str(environment.idx), sign_bit))
                percept = environment.calculate_percept(action)
                label = NNUtility.bitstr_to_narray(percept[1])
                dw_array, db_array = self.nn.backward(activations, label)
                for idx in range(len(self.nn.weights)):
                    self.nn.weights[idx] = self.nn.weights[idx] + dw_array[idx]
                    self.nn.biases[idx] = self.nn.biases[idx] + db_array[idx]
                if step > 0 and turn_number == environment.number_of_turns - 1:
                    if np.log10(step).is_integer():
                        filename = self.__class__.__name__ + "_" + str(self.idx) + "_" + str(self.sign) + "_" + \
                                   str(step) + ".td"
                        path = NNAgent.nn_data_dir.joinpath(filename)
                        self.nn.save_weights(path)

