from typing import Tuple
from python.src.environments.Environment import Environment


class SpecificOne(Environment):
    """Returns reward 1 if the action has a specific 1, otherwise 0."""

    def __init__(self):
        super().__init__()
        self.idx = 4
        self.randomness = False

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns one if the action contains a 1, otherwise 0."""
        if action[1] == "1":
            return "11", "100"
        return "00", "000"
