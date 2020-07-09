import random
from python.src.agents.neural_networks.NeuralNetworkSigmoid import NeuralNetworkSigmoid
from python.src import Utility
from python.src.agents.abstract_classes.Agent import Agent
from python.src.agents.neural_networks import NNUtility
from python.src.environments.abstract_classes.Environment import Environment


class NNAgentSigmoid(Agent):
    """Template for neural network based agents"""

    data_dir = Utility.get_data_path()
    learning_rate = 0.01

    def __init__(self, environment: Environment, hidden_size: [int], activation_name: str = "relu"):
        """Load appropriate parameters depending on environment and learning time"""
        super().__init__(environment)
        input_size = [self.environment.observation_length + self.environment.action_length]
        output_size = [self.environment.reward_length]
        self.nn = NeuralNetworkSigmoid(activation_name, input_size + hidden_size + output_size)
        self.activations = ([], [])

    def calculate_action(self, observation: str) \
            -> str:
        """Feed percept into nn and calculate best activations action. Returns action."""
        action = ""
        reward = -2
        # calculate expected reward by trying every action
        number_of_actions = pow(2, self.environment.action_length)
        if self.seeded_rand_range(10) == 0:
            action_idx = self.seeded_rand_range(number_of_actions)
            action_string = format(action_idx, 'b').zfill(self.environment.action_length)
            nn_input = NNUtility.bitstr_to_narray(observation + action_string)
            nn_output = self.nn.forward(nn_input)
            action = action_string
            self.activations = nn_output
        else:
            for action_idx in range(number_of_actions):
                action_string = format(action_idx, 'b').zfill(self.environment.action_length)
                nn_input = NNUtility.bitstr_to_narray(observation + action_string)
                nn_output = self.nn.forward(nn_input)
                reward_string = NNUtility.narray_to_bitstr(nn_output[1][-1])
                action_reward = Utility.get_reward_from_bitstring(reward_string)
                if action_reward == reward:
                    i = self.seeded_rand_range()
                    if i == 1:
                        action = action_string
                        self.activations = nn_output
                else:
                    if action_reward > reward:
                        action = action_string
                        reward = action_reward
                        self.activations = nn_output
        return action

    def train(self, reward: str):
        """Train agent on received reward"""
        self.nn.backward(self.activations, NNUtility.bitstr_to_narray(reward))

