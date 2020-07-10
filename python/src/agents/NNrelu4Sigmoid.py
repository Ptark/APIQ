from python.src.agents.abstract_classes.NNAgentSigmoid import NNAgentSigmoid
from python.src.environments.abstract_classes.Environment import Environment


class NNrelu4Sigmoid(NNAgentSigmoid):
    """Neural Network based agent with hidden layers"""

    def __init__(self, environment: Environment, seed: int):
        """Initialize NNAgent with given parameters"""
        super().__init__(environment, seed, activation_name="relu", hidden_size=[4])
