from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class CombinationTwo(Environment):
    """Class models a safe with a 2 number combination"""

    number_of_turns = 2
    observation_length = 0
    reward_length = 2
    action_length = 2
    max_average_reward_per_cycle = 0.5

    def __init__(self, sign_bit: str = "0"):
        super().__init__(sign_bit)
        self.lock = False

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns reward 1 if the combination is correctly input over 2 cycles"""
        if not self.lock:
            if action == "11":
                self.lock = True
        else:
            self.lock = False
            if action == "01":
                return '', self.sign_bit + "1"
        return '', self.sign_bit + "0"

