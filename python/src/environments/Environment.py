from abc import ABC, abstractmethod
from typing import Tuple


class Environment(ABC):
    """Abstract class models an environment"""

    @abstractmethod
    def __init__(self):
        self.number_of_turns = 1
        self.idx = 0
        self.randomness = False

    @abstractmethod
    def calculate_percept(self, action: str) \
            -> Tuple[str, str]:
        """To be implemented by child classes.
        Has to return a percept which consists of:
        percept = (observation, reward)"""
        pass
