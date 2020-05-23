from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class SpecificOne(Environment):
    """Returns reward 1 if the action has a specific 1, otherwise 0."""

    observation_length = 0
    reward_length = 2
    action_length = 2

    def __init__(self, sign_bit: str = "0"):
        super().__init__(sign_bit)

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns one if the action contains a 1, otherwise 0."""
        if action[1] == "1":
            return "", self.sign_bit + "1"
        return "", self.sign_bit + "0"

