from typing import Tuple
from python.src.environments.Environment import Environment


class Slide(Environment):
    """Class models a slide which increases reward if you climb the ladder first."""
    def __init__(self):
        self.at_top = False
        super().__init__()
        self.number_of_turns = 2
        self.idx = "2"

    def calculate_percept(self, action: str) \
            -> Tuple[str, str]:
        """Returns a reward of 1/8 + 1/8 if staying at the bottom and 1 if
        climbing up and sliding down"""
        if self.at_top is False:
            if action[0] == "0":
                return "0", "0001"
            else:
                self.at_top = True
                return "1", "0"
        else:
            if action[0] == "0":
                return "1", "0"
            else:
                self.at_top = False
                return "0", "1"

