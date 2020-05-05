from abc import ABC, abstractmethod
from typing import Tuple, Type

from python.src.environments.Environment import Environment


class Agent(ABC):
    """Abstract class models an agent"""

    is_trainable = False

    @abstractmethod
    def __init__(self, environment_class: Type[Environment], sign_bit: str, training_step: int):
        self.environment_class = environment_class
        self.training_step = training_step

    @abstractmethod
    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """To be implemented by child classes
        returns an action as a string of 0s and 1s"""
        pass

    def train(self, label: str):
        """To be implemented by child classes if trainable
        trains an agent in the given environment with the given reward sign_bit"""
        pass

    def reset(self):
        """To be implemented by child classes if necessary
        Resets agent to first turn if needed"""
        pass

    def save(self, sign_bit: str, training_step: int):
        """To be implemented by child classes if trainable
        Saves learning to file"""
        pass
