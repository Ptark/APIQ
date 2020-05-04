from abc import ABC, abstractmethod
from typing import Tuple, Type

from python.src.environments.Environment import Environment


class Agent(ABC):
    """Abstract class models an agent"""

    is_trainable = False

    @abstractmethod
    def __init__(self, environment: Environment, training_step: int):
        self.environment = environment
        self.training_step = training_step

    @abstractmethod
    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """To be implemented by child classes
        returns an action as a string of 0s and 1s"""
        pass

    def train(self):
        """To be implemented by child classes
        trains an agent in the given environment with the given reward sign_bit"""
        pass
