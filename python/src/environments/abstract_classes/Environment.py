from abc import ABC, abstractmethod
import random
from typing import Tuple


class Environment(ABC):
    """Abstract class models an environment"""

    has_randomness = False
    observation_length = 0
    reward_length = 2
    action_length = 1
    max_average_reward_per_cycle = 1

    @abstractmethod
    def __init__(self, sign_bit: str, seed: int = 1):
        self.sign_bit = sign_bit
        self.seed = seed

    @abstractmethod
    def calculate_percept(self, action: str) \
            -> Tuple[str, str]:
        """To be implemented by child classes.
        Has to return a percept which consists of:
        percept = (observation, reward)"""
        pass

    def get_random_bit(self) -> int:
        """Returns a seeded random bit and increments internal seed"""
        random.seed(self.seed)
        self.seed += 1
        return random.randint(0, 1)
