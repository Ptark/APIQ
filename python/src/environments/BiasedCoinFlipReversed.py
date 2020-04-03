from typing import Tuple
import random

from python.src.environments.Environment import Environment


class BiasedCoinFlipReversed(Environment):
    """Class models a biased 30% coin flip."""

    def __init__(self):
        super().__init__()

    def calculate_percept(self, prediction: Tuple[int, int, int, int]) \
            -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Takes a prediction, throws a coin with 30% chance for heads (0) and returns observation and reward"""
        outcome = random.randint(0, 9)
        if outcome >= 3:
            outcome = 1
        else:
            outcome = 0
        if prediction[0] == outcome:
            return (outcome, 0, 0, 0), (1, 0, 0, 0)
        else:
            return (outcome, 0, 0, 0), (1, 1, 0, 0)
