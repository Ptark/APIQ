from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class CombinationThree(Environment):
    """Class models a safe with a 3 number combination"""

    observation_length = 0
    reward_length = 2
    action_length = 2
    max_average_reward_per_cycle = 1/3

    def __init__(self, sign_bit: str = "0", seed: int = 1):
        super().__init__(sign_bit, seed)
        self.lock_one = False
        self.lock_two = False

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns reward 1 if the combination is correctly input over 3 cycles"""
        if not self.lock_one:
            if action == "10":
                self.lock_one = True
        elif not self.lock_two:
            if action == "00":
                self.lock_two = True
            else:
                self.lock_one = False
        else:
            self.lock_one = False
            self.lock_two = False
            if action == "11":
                return '', self.sign_bit + "1"
        return '', self.sign_bit + "0"

