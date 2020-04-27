import numpy as np
from typing import Tuple

from python.src import Utility
from python.src.agents.Agent import Agent
from python.src.agents.neural_networks import NNUtility
from python.src.agents.neural_networks.NeuralNetwork import NeuralNetwork
from python.src.environments import Environment


class NNAgent(Agent):
    """Template for neural network based agents"""

    def __init__(self, activation_name: str, size: [int], training_steps: int):
        """Load appropriate parameters depending on environment and learning time"""
        super().__init__()
        self.nn = NeuralNetwork(activation_name, size)
        self.training_steps = training_steps

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Calculates an action as a string of 0s and 1s from the neural network."""
        observation = percept[0]
        action = ""
        if self.turn_counter == 0:
            self.idx = int(percept[0])
            observation = "1111"
            self.sign = 1 - 2 * (int(percept[1]) == 1)
            # load parameters for idx and training steps and sign
        reward = -2
        for action_idx in range(15):
            action_string = format(str(action_idx), 'b')
            nn_in = np.fromstring(observation + action_string, float)
            nn_out = self.nn.forward(nn_in)[1][-1]
            bit_string = NNUtility.narray_to_bitstr(nn_out)
            action_reward = self.sign * Utility.get_reward_from_bitstring(bit_string)
            if action_reward >= reward:
                action = action_string
                reward = action_reward
        self.turn_counter += 1
        return action

    def train(self, environment: Environment) -> bool:
        pass



