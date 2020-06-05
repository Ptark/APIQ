from python.src.agents.abstract_classes.KerasNNAgent import KerasNNAgent
from python.src.environments.abstract_classes.Environment import Environment


class KerasNNrelu(KerasNNAgent):
    """Keras Neural Network based agent with no hidden layers"""
    def __init__(self, environment: Environment):
        """Initialize KerasNNAgent with given parameters"""
        super().__init__(environment, activation_name="relu", hidden_size=[])
