from typing import Tuple
import random

from python.src.environments.abstract_classes.Environment import Environment


class BiasedCoinFlipR(Environment):
    """Class models a biased 25% coin flip with reversed reward."""

    has_randomness = True
    observation_length = 1
    reward_length = 2
    action_length = 1
    sign_bit = "1"

    def __init__(self):
        super().__init__()

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Takes a prediction, throws a coin with 25% chance for heads (0) and returns observation and reward"""
        outcome = random.randint(0, 1) + random.randint(0, 1)
        outcome = 0 if outcome == 0 else 1
        if int(action[-1]) == outcome:
            return str(outcome), BiasedCoinFlipR.sign_bit + "1"
        else:
            return str(outcome), BiasedCoinFlipR.sign_bit + "0"
