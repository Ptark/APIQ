from typing import Tuple

import random

from python.src.environments.Environment import Environment


class DoubleCoinFlip(Environment):
    """Class models a double coin flip."""

    randomness = True
    observation_length = 2
    reward_length = 2
    action_length = 2

    def __init__(self, sign_bit: str):
        super().__init__(sign_bit)

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Takes a prediction, throws two coins and returns observation and reward"""
        outcome_one = random.randint(0, 1)
        outcome_two = random.randint(0, 1)
        prediction = 0
        prediction += int(action[-1])
        prediction += int(action[-2])
        if prediction == outcome_one + outcome_two:
            return str(outcome_two) + str(outcome_one), self.sign_bit + "1"
        else:
            return str(outcome_two) + str(outcome_one), self.sign_bit + "0"
