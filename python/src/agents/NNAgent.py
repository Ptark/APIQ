import random
from typing import Tuple, Type

from python.src import Utility
from python.src.agents.Agent import Agent
from python.src.agents.neural_networks import NNUtility
from python.src.agents.neural_networks.NeuralNetwork import NeuralNetwork
from python.src.environments.Environment import Environment


class NNAgent(Agent):
    """Template for neural network based agents"""

    is_trainable = True
    training_data_dir = Utility.get_resources_path().joinpath('data/training_data')
    learning_rate = 0.01

    def __init__(self, environment_class: Type[Environment], sign_bit: str, training_step: int, activation_name: str, hidden_size: [int]):
        """Load appropriate parameters depending on environment and learning time"""
        super().__init__(environment_class, sign_bit, training_step)
        input_size = [self.environment_class.observation_length + self.environment_class.action_length]
        output_size = [self.environment_class.reward_length]
        self.nn = NeuralNetwork(activation_name, input_size + hidden_size + output_size)
        self.activations = ([], [])
        if self.training_step > 0:
            filename = self.__class__.__name__ + "_" + self.environment_class.__name__ + sign_bit + "_" + str(self.training_step) + ".apiq"
            path = NNAgent.training_data_dir.joinpath(filename)
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

    def save(self, sign_bit: str, training_step: int):
        """Save weights to file"""
        filename = self.__class__.__name__ + "_" + self.environment_class.__name__ + sign_bit + "_" + str(
            training_step) + ".apiq"
        path = NNAgent.training_data_dir.joinpath(filename)
        self.nn.save_weights(path)

