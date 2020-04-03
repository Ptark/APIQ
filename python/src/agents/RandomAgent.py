import random
from typing import Tuple

from python.src.agents.Agent import Agent


class RandomAgent(Agent):
    """Models an agent which takes random actions independent of percepts"""

    def __init__(self):
        super().__init__()

    def calculate_action(self, percept: Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]) \
            -> Tuple[int, int, int, int]:
        """returns a random action"""
        return random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)
