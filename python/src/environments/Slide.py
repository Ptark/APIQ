from typing import Tuple
from python.src.environments.Environment import Environment


class Slide(Environment):
    """Class models a slide which increases reward if you climb the ladder first."""
    def __init__(self):
        self.at_bottom = True
        super().__init__()
        self.turns = 2
        self.idx = (0, 1, 0, 0)

    def calculate_percept(self, action: Tuple[int, int, int, int]) \
            -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Takes a prediction, throws a coin with 30% chance for heads (0) and returns observation and reward"""
        if self.at_bottom is True:
            if action[0] == 0:
                return (0, 0, 0, 0), (0, 0, 0, 1)
            else:
                self.at_bottom = False
                return (1, 0, 0, 0), (0, 0, 0, 0)
        else:
            self.at_bottom = True
            return (0, 0, 0, 0), (1, 0, 0, 0)

