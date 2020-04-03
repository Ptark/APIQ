from typing import Tuple

import random

from python.src.environments.Environment import Environment


class DoubleCoinFlip(Environment):
    """Class models a double coin flip."""

    def __init__(self):
        super().__init__()

    def calculate_percept(self, prediction: Tuple[int, int, int, int]) \
            -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Takes a prediction, throws two coins and returns observation and reward
        The returned percept is built as follows:
        (observation bits, reward bits(Sign, 1, 1/2, 1/4))"""
        outcome_one = random.randint(0, 1)
        outcome_two = random.randint(0, 1)
        if prediction[0] + prediction[1] == outcome_one + outcome_two:
            return (outcome_one, outcome_two, 0, 0), (1, 0, 0, 0)
        else:
            return (outcome_one, outcome_two, 0, 0), (0, 0, 0, 0)

