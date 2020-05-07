from typing import Tuple

from python.src.environments.AnyOne import AnyOne
from python.src.environments.Environment import Environment


class AnyOneR(Environment):
    """Returns 0 if the action contains a 1, otherwise 1."""

    observation_length = 0
    reward_length = 2
    action_length = 2
    sign_bit = "1"

    def __init__(self):
        super().__init__()

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns one if the action contains a 1, otherwise 0."""
        if "1" in action:
            return "", AnyOneR.sign_bit + "1"
        return "", AnyOneR.sign_bit + "0"

