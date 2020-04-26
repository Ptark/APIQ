from abc import ABC, abstractmethod
from typing import Tuple


class Agent(ABC):
    """Abstract class models an agent"""

    @abstractmethod
    def __init__(self):
        self.idx = "0"
        self.sign = "0"
        self.turn_counter = 0

    @abstractmethod
    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """To be implemented by child classes
        returns an action as a string of 0s and 1s"""
        pass
