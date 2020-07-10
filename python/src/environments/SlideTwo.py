from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class SlideTwo(Environment):
    """Class models a slide which increases reward if you climb the ladder first.
    You can climb twice"""

    observation_length = 2
    reward_length = 5
    action_length = 1
    max_average_reward_per_cycle = 1/3

    def __init__(self, sign_bit: str = "0", seed: int = 1):
        super().__init__(sign_bit, seed)
        self.at_bottom = True
        self.at_middle = False

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns a reward of 1/8 + 1/8 if staying at the bottom and 1/2 if
        climbing up and sliding down and 1 if climbing twice and sliding down"""
        if self.at_bottom:
            if action == "0":
                return "00", self.sign_bit + "0001"
            self.at_bottom = False
            self.at_middle = True
            return "01", self.sign_bit + "0000"
        elif self.at_middle:
            self.at_middle = False
            if action == "0":
                self.at_bottom = True
                return "00", self.sign_bit + "0100"
            if action == "1":
                return "10", self.sign_bit + "0000"
        else:
            self.at_bottom = True
            return "00", self.sign_bit + "1000"

