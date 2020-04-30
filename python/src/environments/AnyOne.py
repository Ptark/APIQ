from typing import Tuple
from python.src.environments.Environment import Environment


class AnyOne(Environment):
    """Returns one if the action contains a 1, otherwise 0."""

    def __init__(self):
        super().__init__()
        self.idx = 3
        self.randomness = False

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns one if the action contains a 1, otherwise 0."""
        if "1" in action:
            return "11", "100"
        return "00", "000"

