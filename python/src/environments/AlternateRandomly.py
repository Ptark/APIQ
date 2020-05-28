import random
from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class AlternateRandomly(Environment):
    """Class models an environment where the correct action 0 or 1 changes with
    a chance of 6.25%."""

    observation_length = 0
    reward_length = 2
    action_length = 1
    max_average_reward_per_cycle = 1
    has_randomness = True

    def __init__(self, sign_bit: str = "0"):
        super().__init__(sign_bit)
        self.zero = True

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns 1 if the correct action is sent. The correct action flips with 6.25% chance"""
        flip = random.randint(0, 1) + random.randint(0, 1) + random.randint(0, 1) + random.randint(0, 1)
        if flip == 0:
            self.zero = not self.zero
        if (self.zero and action == "0") or (not self.zero and action == "1"):
            return "", self.sign_bit + "1"
        return "", self.sign_bit + "0"
