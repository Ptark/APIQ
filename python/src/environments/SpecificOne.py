from typing import Tuple
from python.src.environments.Environment import Environment


class SpecificOne(Environment):
    """Returns reward 1 if the action has a specific 1, otherwise 0."""

    observation_length = 0
    reward_length = 2
    action_length = 2

    def __init__(self, sign: str):
        super().__init__(sign)

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns one if the action contains a 1, otherwise 0."""
        if action[1] == "1":
            return "", self.sign + "1"
        return "", self.sign + "0"

