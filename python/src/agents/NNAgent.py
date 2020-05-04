import random
import numpy as np
from typing import Tuple, List, Type

from python.src import Utility
from python.src.agents.Agent import Agent
from python.src.agents.neural_networks import NNUtility
from python.src.agents.neural_networks.NeuralNetwork import NeuralNetwork
from python.src.environments.Environment import Environment


class NNAgent(Agent):
    """Template for neural network based agents"""

    training_steps = 1000
    nn_data_dir = Utility.get_resources_path().joinpath('resources/agents/nn_data')
    learning_rate = 0.01

    def __init__(self, environment: Environment, training_step: int, activation_name: str, hidden_size: [int]):
        """Load appropriate parameters depending on environment and learning time"""
        super().__init__(environment, training_step)
        input_size = [self.environment.observation_length + self.environment.action_length]
        output_size = [self.environment.reward_length]
        self.nn = NeuralNetwork(activation_name, input_size + hidden_size + output_size)
        if self.training_step > 0:
            filename = self.__class__.__name__ + "_" + self.environment.__class__.__name__ + str(self.environment.sign) \
                       + "_" + str(self.training_step) + ".td"
            path = NNAgent.nn_data_dir.joinpath(filename)
            self.nn.load_weights(path)

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Calculates an action as a string of 0s and 1s from the neural network."""
        return self.calculate_activations_and_action(percept)[1]

    def calculate_activations_and_action(self, percept: Tuple[str, str]) \
            -> Tuple[Tuple[List[np.ndarray], List[np.ndarray]], str]:
        """Feed percept into nn and returns activations and action"""
        action = ""
        activations = []
        reward = -2
        # calculate expected reward by trying every action
        number_of_actions = pow(2, self.environment.action_length)
        for action_idx in range(number_of_actions):
            action_string = format(action_idx, 'b').zfill(self.environment.action_length)
            nn_input = NNUtility.bitstr_to_narray(percept[0] + action_string)
            nn_output = self.nn.forward(nn_input)
            bit_string = NNUtility.narray_to_bitstr(nn_output[1][-1])
            action_reward = Utility.get_reward_from_bitstring(bit_string)
            if action_reward == reward:
                i = random.randint(0, 1)
                if i == 1:
                    action = action_string
                    activations = nn_output
            else:
                if action_reward > reward:
                    action = action_string
                    reward = action_reward
                    activations = nn_output
        self.turn_counter += 1
        return activations, action

    def train(self):
        """Train agent in environment training_steps times"""
        for step in range(NNAgent.training_steps):
            self.turn_counter = 0
            environment = environment_class()
            percept = (str(environment.idx), sign_bit)
            for turn_number in range(environment.number_of_turns):
                activations, action = self.calculate_activations_and_action(percept)
                percept = environment.calculate_percept(action)
                label = NNUtility.bitstr_to_narray(percept[1])
                self.nn.backward(activations, label)
                if turn_number == environment.number_of_turns - 1:
                    if step % 100 == 0:
                        filename = self.__class__.__name__ + "_" + str(self.idx) + "_" + str(self.sign) + "_" + \
                                   str(step) + ".td"
                        path = NNAgent.nn_data_dir.joinpath(filename)
                        self.nn.save_weights(path)

