from abc import ABC, abstractmethod
from typing import Tuple


class Environment(ABC):
    """Abstract class models an environment"""

    @abstractmethod
    def __init__(self):
        self.turns = 1
        self.idx = (0, 0, 0, 0)

    @abstractmethod
    def calculate_percept(self, action: Tuple[int, int, int, int]) \
            -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """To be implemented by child classes.
        Has to return a percept which consists of:
        percept = (observation, reward(1, 1/2, 1/4, 1/8))"""
        pass
