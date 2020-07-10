from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class DoubleCoinFlip(Environment):
    """Class models a double coin flip."""

    has_randomness = True
    observation_length = 2
    reward_length = 2
    action_length = 2

    def __init__(self, sign_bit: str = "0", seed: int = 1):
        super().__init__(sign_bit, seed)

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Takes a prediction, throws two coins and returns observation and reward"""
        outcome_one = self.get_random_bit()
        outcome_two = self.get_random_bit()
        prediction = 0
        prediction += int(action[-1])
        prediction += int(action[-2])
        if prediction == outcome_one + outcome_two:
            return str(outcome_two) + str(outcome_one), self.sign_bit + "1"
        else:
            return str(outcome_two) + str(outcome_one), self.sign_bit + "0"
