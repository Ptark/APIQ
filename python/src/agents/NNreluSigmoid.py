from python.src.agents.abstract_classes.NNAgentSigmoid import NNAgentSigmoid
from python.src.environments.abstract_classes.Environment import Environment


class NNreluSigmoid(NNAgentSigmoid):
    """Neural Network based agent with no hidden layers"""

    def __init__(self, environment: Environment, seed: int):
        """Initialize NNAgent with given parameters"""
        super().__init__(environment, seed, hidden_size=[], activation_name="relu")
