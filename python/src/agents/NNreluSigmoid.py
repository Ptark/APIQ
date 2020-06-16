from agents.abstract_classes.NNAgentSigmoid import NNAgentSigmoid
from python.src.environments.abstract_classes.Environment import Environment


class NNreluSigmoid(NNAgentSigmoid):
    """Neural Network based agent with no hidden layers"""

    def __init__(self, environment: Environment):
        """Initialize NNAgent with given parameters"""
        super().__init__(environment, hidden_size=[], activation_name="relu")
