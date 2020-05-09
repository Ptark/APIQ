import random
from pathlib import Path
from typing import Tuple, Type

from python.src import Utility
from python.src.agents.abstract_classes.TrainableAgent import TrainableAgent
from python.src.agents.neural_networks import NNUtility
from python.src.agents.neural_networks.NeuralNetwork import NeuralNetwork
from python.src.environments.abstract_classes.Environment import Environment


class NNAgent(TrainableAgent):
    """Template for neural network based agents"""

    is_trainable = True
    data_dir = Utility.get_data_path()
    learning_rate = 0.01
    has_randomness = True

    def __init__(self, environment_class: Type[Environment], hidden_size: [int], activation_name: str = "relu", path: Path = ''):
        """Load appropriate parameters depending on environment and learning time"""
        super().__init__(environment_class, path)
        input_size = [self.environment_class.observation_length + self.environment_class.action_length]
        output_size = [self.environment_class.reward_length]
        self.nn = NeuralNetwork(activation_name, input_size + hidden_size + output_size)
        self.activations = ([], [])
        if path != '':
            self.nn.load_weights(path)

    def calculate_action(self, percept: Tuple[str, str]) \
            -> str:
        """Feed percept into nn and calculate best activations action. Returns action."""
        action = ""
        reward = -2
        # calculate expected reward by trying every action
        number_of_actions = pow(2, self.environment_class.action_length)
        for action_idx in range(number_of_actions):
            action_string = format(action_idx, 'b').zfill(self.environment_class.action_length)
            nn_input = NNUtility.bitstr_to_narray(percept[0] + action_string)
            nn_output = self.nn.forward(nn_input)
            bit_string = NNUtility.narray_to_bitstr(nn_output[1][-1])
            action_reward = Utility.get_reward_from_bitstring(bit_string)
            if action_reward == reward:
                i = random.randint(0, 1)
                if i == 1:
                    action = action_string
                    self.activations = nn_output
            else:
                if action_reward > reward:
                    action = action_string
                    reward = action_reward
                    self.activations = nn_output
        return action

    def train(self, reward_bits: str):
        """Train agent on received reward"""
        self.nn.backward(self.activations, NNUtility.bitstr_to_narray(reward_bits))

    def save(self, path: Path = ''):
        """Save weights to file"""
        self.nn.save_weights(path)

