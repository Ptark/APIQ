from typing import Tuple
from python.src.environments.Environment import Environment


class AnyOne(Environment):
    """Returns one if the action contains a 1, otherwise 0."""

    observation_length = 0
    reward_length = 2
    action_length = 2

    def __init__(self, sign_bit: str = "0"):
        super().__init__(sign_bit)

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns one if the action contains a 1, otherwise 0."""
        if "1" in action:
            return "", self.sign_bit + "1"
        return "", self.sign_bit + "0"

