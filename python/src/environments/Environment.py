from abc import ABC, abstractmethod
from typing import Tuple


class Environment(ABC):
    """Abstract class models an environment"""

    number_of_turns = 1
    has_randomness = False
    observation_length = 0
    reward_length = 2
    action_length = 1
    sign_bit = "0"

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calculate_percept(self, action: str) \
            -> Tuple[str, str]:
        """To be implemented by child classes.
        Has to return a percept which consists of:
        percept = (observation, reward)"""
        pass
