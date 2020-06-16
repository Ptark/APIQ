from agents.abstract_classes.NNAgentSigmoid import NNAgentSigmoid
from python.src.environments.abstract_classes.Environment import Environment


class NNrelu4Sigmoid(NNAgentSigmoid):
    """Neural Network based agent with hidden layers"""

    def __init__(self, environment: Environment):
        """Initialize NNAgent with given parameters"""
        super().__init__(environment, activation_name="relu", hidden_size=[4])
