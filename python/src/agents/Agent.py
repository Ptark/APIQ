from abc import ABC, abstractmethod
from typing import Tuple, Type

from python.src.environments.Environment import Environment


class Agent(ABC):
    """Abstract class models an agent"""

    training_steps = 1

    @abstractmethod
    def __init__(self, training_step: int = 0):
        self.idx = 0
        self.sign = 0
        self.turn_counter = 0
        self.training_step = training_step
        self.is_initialized = False

    @abstractmethod
    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """To be implemented by child classes
        returns an action as a string of 0s and 1s"""
        pass

    def train(self, environment: Type[Environment], sign_bit: str):
        """To be implemented by child classes
        trains an agent in the given environment with the given reward sign_bit"""
        pass
