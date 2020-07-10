from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class Button(Environment):

    observation_length = 0
    reward_length = 2
    action_length = 1

    def __init__(self, sign_bit: str = "0", seed: int = 1):
        super().__init__(sign_bit, seed)

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns reward 1 if the action is 1"""
        if action == "1":
            return '', self.sign_bit + "1"
        return '', self.sign_bit + "0"

