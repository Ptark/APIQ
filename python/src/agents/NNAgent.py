from typing import Tuple

from python.src.agents.Agent import Agent
from python.src.environments import Environment


class NNAgent(Agent):
    """Template for neural network based agents"""

    def __init__(self):
        """Load appropriate parameters depending on environment and learning time"""
        super().__init__()
        # set environment idx
        # load appropriate parameters

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """To be implemented by child classes.
        Calculates an action as a string of 0s and 1s from the neural network."""

    def train(self, environment: Environment) -> bool:
        pass



