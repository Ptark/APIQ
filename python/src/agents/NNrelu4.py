from python.src.agents.abstract_classes.NNAgent import NNAgent
from python.src.environments.abstract_classes.Environment import Environment


class NNrelu4(NNAgent):
    """Neural Network based agent with hidden layers"""

    def __init__(self, environment: Environment):
        """Initialize NNAgent with given parameters"""
        super().__init__(environment, activation_name="relu", hidden_size=[4])
