from abc import ABC, abstractmethod
from typing import Tuple


class Agent(ABC):
    """Abstract class models an agent"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calculate_action(self, percept: Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]])\
            -> Tuple[int, int, int, int]:
        """To be implemented by child classes
        returns an action which is a tuple with 4 bits/integers"""
        pass
