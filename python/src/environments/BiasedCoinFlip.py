from typing import Tuple
import random

from python.src.environments.Environment import Environment


class BiasedCoinFlip(Environment):
    """Class models a biased 30% coin flip."""

    def __init__(self):
        super().__init__()
        self.idx = 0
        self.randomness = True

    def calculate_percept(self, prediction: str) -> Tuple[str, str]:
        """Takes a prediction, throws a coin with 30% chance for heads (0) and returns observation and reward"""
        outcome = random.randint(0, 9)
        outcome = 1 if outcome >= 3 else 0
        if int(prediction[-1]) == outcome:
            return "0" + str(outcome), "100"
        else:
            return "0" + str(outcome), "000"
