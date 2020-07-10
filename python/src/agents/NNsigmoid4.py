from python.src.agents.abstract_classes.NNAgent import NNAgent
from python.src.environments.abstract_classes.Environment import Environment


class NNsigmoid4(NNAgent):
    """Neural Network based agent with a hidden layer of width 4"""

    def __init__(self, environment: Environment, seed: int):
        """Initialize NNAgent with given parameters"""
        super().__init__(environment, seed, activation_name="sigmoid", hidden_size=[4])
