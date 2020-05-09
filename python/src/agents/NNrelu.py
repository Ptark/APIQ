from pathlib import Path
from typing import Type

from python.src.agents.abstract_classes.NNAgent import NNAgent
from python.src.environments.abstract_classes.Environment import Environment


class NNrelu(NNAgent):
    """Neural Network based agent with no hidden layers"""

    def __init__(self, environment_class: Type[Environment], path: Path = ''):
        """Initialize NNAgent with given parameters"""
        super().__init__(environment_class, path=path, hidden_size=[], activation_name="relu")
