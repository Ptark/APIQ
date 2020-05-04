from typing import Tuple
from python.src.environments.Environment import Environment


class Slide(Environment):
    """Class models a slide which increases reward if you climb the ladder first."""

    number_of_turns = 2
    observation_length = 1
    reward_length = 4
    action_length = 1

    def __init__(self, sign: str):
        self.at_top = False
        super().__init__(sign)

    def calculate_percept(self, action: str) \
            -> Tuple[str, str]:
        """Returns a reward of 1/4 + 1/4 if staying at the bottom and 1 if
        climbing up and sliding down"""
        if self.at_top is False:
            if action[-1] == "0":
                return "0", self.sign + "001"
            else:
                self.at_top = True
                return "1", self.sign + "000"
        else:
            if action[-1] == "0":
                return "1", self.sign + "000"
            else:
                self.at_top = False
                return "0", self.sign + "100"

