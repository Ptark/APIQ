from abc import ABC, abstractmethod
from typing import Tuple


class Agent(ABC):
    """Abstract class models an agent"""

    @abstractmethod
    def __init__(self):
        self.idx = 0
        self.sign = 1
        self.turn = 0

    @abstractmethod
    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """To be implemented by child classes
        returns an action which is a tuple with 4 bits/integers"""
        pass
