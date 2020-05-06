from typing import Type

from python.src.agents.NNAgent import NNAgent
from python.src.environments.Environment import Environment


class NNrelu4(NNAgent):
    """Neural Network based agent with hidden layers"""

    def __init__(self, environment_class: Type[Environment], sign_bit: str = "0", training_step: int = 0):
        """Initialize NNAgent with given parameters"""
        super().__init__(environment_class, sign_bit, training_step, activation_name="relu", hidden_size=[4])
