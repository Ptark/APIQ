from typing import Tuple
import random

from python.src.environments.Environment import Environment


class BiasedCoinFlip(Environment):
    """Class models a biased 30% coin flip."""

    randomness = True
    observation_length = 1
    reward_length = 2
    action_length = 1

    def __init__(self, sign):
        super().__init__(sign)

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Takes a prediction, throws a coin with 30% chance for heads (0) and returns observation and reward"""
        outcome = random.randint(0, 9)
        outcome = 1 if outcome >= 3 else 0
        if int(action[-1]) == outcome:
            return str(outcome), self.sign + "1"
        else:
            return str(outcome), self.sign + "0"
