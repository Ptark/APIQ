from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class Alternate(Environment):

    observation_length = 0
    reward_length = 2
    action_length = 1

    def __init__(self, sign_bit: str = "0"):
        super().__init__(sign_bit)
        self.action = "1"

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns reward 1 if the action is different from the last one, else 0"""
        if self.action == action:
            return '', self.sign_bit + "0"
        else:
            self.action = action
            return '', self.sign_bit + "1"

