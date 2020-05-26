from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class Ramp(Environment):
    """Class models a slide which increases reward if you climb the ladder first."""

    observation_length = 1
    reward_length = 5
    action_length = 1
    max_average_reward_per_cycle = 0.5

    def __init__(self, sign_bit: str = "0"):
        super().__init__(sign_bit)
        self.at_bottom = True

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns a reward of 1/8 + 1/8 if staying at the bottom and 1 if
        climbing up and sliding down"""
        if self.at_bottom:
            if action == "0":
                return "0", self.sign_bit + "0001"
            else:
                self.at_bottom = False
                return "1", self.sign_bit + "0000"
        else:
            self.at_bottom = True
            return "0", self.sign_bit + "1000"
