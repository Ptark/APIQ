from typing import Tuple
import random

from python.src.environments.Environment import Environment


class BiasedCoinFlip(Environment):
    """Class models a biased 25% coin flip."""

    has_randomness = True
    observation_length = 1
    reward_length = 2
    action_length = 1

    def __init__(self, sign_bit: str = "0"):
        super().__init__(sign_bit)

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Takes a prediction, throws a coin with 30% chance for heads (0) and returns observation and reward"""
        outcome = random.randint(0, 1) + random.randint(0, 1)
        outcome = 0 if outcome == 0 else 1
        if int(action[-1]) == outcome:
            return str(outcome), self.sign_bit + "1"
        else:
            return str(outcome), self.sign_bit + "0"
