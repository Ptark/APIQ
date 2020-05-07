from typing import Tuple
from python.src.environments.Environment import Environment


class SpecificOneR(Environment):
    """Returns reward 1 if the action has a specific 1, otherwise 0."""

    observation_length = 0
    reward_length = 2
    action_length = 2
    sign_bit = "1"

    def __init__(self):
        super().__init__()

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns one if the action contains a 1, otherwise 0."""
        if action[1] == "1":
            return "", SpecificOneR.sign_bit + "1"
        return "", SpecificOneR.sign_bit + "0"

