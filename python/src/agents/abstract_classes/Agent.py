from abc import ABC, abstractmethod
from typing import Tuple, Type

from python.src.environments.abstract_classes.Environment import Environment


class Agent(ABC):
    """Abstract class models an agent"""

    @abstractmethod
    def __init__(self, environment: Environment):
        self.environment = environment

    @abstractmethod
    def calculate_action(self, observation: str) -> str:
        """To be implemented by child classes
        returns an action as a string of 0s and 1s"""
        pass

    def train(self, reward: str):
        """Trains agent on observation and reward if trainable"""
        pass

