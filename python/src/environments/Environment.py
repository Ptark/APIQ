from abc import ABC, abstractmethod
from typing import Tuple


class Environment(ABC):
    """Abstract class models an environment"""

    number_of_turns = 1
    randomness = False
    observation_length = 0
    reward_length = 2
    action_length = 1

    @abstractmethod
    def __init__(self, sign_bit: str):
        self.sign_bit = sign_bit

    @abstractmethod
    def calculate_percept(self, action: str) \
            -> Tuple[str, str]:
        """To be implemented by child classes.
        Has to return a percept which consists of:
        percept = (observation, reward)"""
        pass
