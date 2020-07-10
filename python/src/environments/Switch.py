from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class Switch(Environment):

    observation_length = 1
    reward_length = 2
    action_length = 1

    def __init__(self, sign_bit: str = "0", seed: int = 1):
        super().__init__(sign_bit, seed)
        self.turn = 0
        self.zero = True

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns reward 1 for 0 in odd rounds and reward 1 for 1 in even rounds.
        Reward is negated if sign_bit is 1. Observation indicates status."""
        if self.zero:
            self.zero = False
            if action == "0":
                return "0", self.sign_bit + "1"
            else:
                return "0", self.sign_bit + "0"
        else:
            self.zero = True
            if action == "1":
                return "1", self.sign_bit + "1"
            else:
                return "1", self.sign_bit + "0"

