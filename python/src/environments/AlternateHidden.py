from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class AlternateHidden(Environment):

    number_of_turns = 2
    observation_length = 0
    reward_length = 2
    action_length = 1

    def __init__(self, sign_bit: str = "0"):
        super().__init__(sign_bit)
        self.turn = 0
        self.zero = True

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns reward 1 for 0 in odd rounds and reward 1 for 1 in even rounds.
        Reward is negated if sign_bit is 1"""
        if self.zero:
            self.zero = False
            if action == "0":
                return "", self.sign_bit + "1"
            else:
                return "", self.sign_bit + "0"
        else:
            self.zero = True
            if action == "1":
                return "", self.sign_bit + "1"
            else:
                return "", self.sign_bit + "0"

