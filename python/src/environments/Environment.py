from abc import ABC, abstractmethod
from typing import Tuple


class Environment(ABC):
    """Abstract class models an environment"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calculate_percept(self, action: Tuple[int, int, int, int]) \
            -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """To be implemented by child classes.
        Has to return a percept which consists of:
        percept = (observation, reward(sign, 1, 1/2, 1/4))"""
        pass
