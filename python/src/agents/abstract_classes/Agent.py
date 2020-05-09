from abc import ABC, abstractmethod
from typing import Tuple, Type

from python.src.environments.abstract_classes.Environment import Environment


class Agent(ABC):
    """Abstract class models an agent"""

    has_randomness = False

    @abstractmethod
    def __init__(self, environment_class: Type[Environment]):
        self.environment_class = environment_class

    @abstractmethod
    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """To be implemented by child classes
        returns an action as a string of 0s and 1s"""
        pass

    def reset(self):
        """To be implemented by child classes if necessary
        Resets agent to first turn if needed"""
        pass
