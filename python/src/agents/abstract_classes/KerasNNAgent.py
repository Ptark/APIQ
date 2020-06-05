import random

import numpy as np

from python.src import Utility
from python.src.agents.abstract_classes.Agent import Agent
from python.src.agents.neural_networks import NNUtility
from python.src.environments.abstract_classes.Environment import Environment
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense


class KerasNNAgent(Agent):
    """Template for keras neural network based agents"""

    is_trainable = True
    data_dir = Utility.get_data_path()
    learning_rate = 0.01

    def __init__(self, environment: Environment, hidden_size: [int], activation_name: str = "relu"):
        """Load appropriate parameters depending on environment and learning time"""
        super().__init__(environment)
        input_size = self.environment.observation_length + self.environment.action_length
        output_size = self.environment.reward_length
        self.nn = Sequential()
        self.nn.add(Input(shape=(input_size,)))
        for hidden_layer_size in hidden_size:
            self.nn.add(Dense(hidden_layer_size, activation=activation_name))
        self.nn.add(Dense(output_size, activation=activation_name))
        self.nn.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.nn_input = np.zeros((1, self.environment.observation_length + self.environment.action_length))

    def calculate_action(self, observation: str) \
            -> str:
        """Feed percept into nn and calculate best activations action. Returns action."""
        action = ""
        reward = -2
        # calculate expected reward by trying every action
        number_of_actions = pow(2, self.environment.action_length)
        if random.randrange(10) == 0:
            action_idx = random.randrange(number_of_actions)
            action_string = format(action_idx, 'b').zfill(self.environment.action_length)
            nn_input = NNUtility.bitstr_to_narray(observation + action_string).T
            action = action_string
            self.nn_input = nn_input
        else:
            for action_idx in range(number_of_actions):
                action_string = format(action_idx, 'b').zfill(self.environment.action_length)
                nn_input = NNUtility.bitstr_to_narray(observation + action_string).T
                nn_output = self.nn.predict(nn_input, batch_size=1)
                reward_string = NNUtility.narray_to_bitstr(nn_output)
                action_reward = Utility.get_reward_from_bitstring(reward_string)
                if action_reward == reward:
                    i = random.randint(0, 1)
                    if i == 1:
                        action = action_string
                        self.nn_input = nn_input
                else:
                    if action_reward > reward:
                        action = action_string
                        reward = action_reward
                        self.nn_input = nn_input
        return action

    def train(self, reward: str):
        """Train agent on received reward"""
        self.nn.fit(self.nn_input, NNUtility.bitstr_to_narray(reward).T, batch_size=1, epochs=1, verbose=0)

