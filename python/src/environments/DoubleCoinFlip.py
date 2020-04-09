from typing import Tuple

import random

from python.src.environments.Environment import Environment


class DoubleCoinFlip(Environment):
    """Class models a double coin flip."""

    def __init__(self):
        super().__init__()
        self.idx = "1"

    def calculate_percept(self, prediction: str) -> Tuple[str, str]:
        """Takes a prediction, throws two coins and returns observation and reward"""
        outcome_one = random.randint(0, 1)
        outcome_two = random.randint(0, 1)
        if int(prediction[0]) + int(prediction[1]) == outcome_one + outcome_two:
            return str(outcome_two) + str(outcome_one), "1"
        else:
            return str(outcome_two) + str(outcome_one), "0"

