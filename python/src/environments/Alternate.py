from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class Alternate(Environment):
    """Returns 0.5 for 0 in the first round and 0.5 for 1 in the second."""

    number_of_turns = 2
    observation_length = 1
    reward_length = 3
    action_length = 1

    def __init__(self):
        super().__init__()
        self.turn = 0

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns 0.5 for 0 in the first round and 0.5 for 1 in the second."""
        if self.turn == 0:
            self.turn += 1
            if action == "0":
                return "0", Alternate.sign_bit + "01"
            else:
                return "0", Alternate.sign_bit + "00"
        else:
            if action == "1":
                return "1", Alternate.sign_bit + "01"
            else:
                return "1", Alternate.sign_bit + "00"

