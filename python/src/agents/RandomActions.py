from typing import Tuple
import random

from python.src.agents.Agent import Agent
from python.src.environments.Environment import Environment


class RandomActions(Agent):
    """Models an agent which takes handcrafted actions depending on environments"""

    training_steps = 1

    def __init__(self, environment: Environment, training_step: int = 0):
        super().__init__(environment, training_step)

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Returns handcrafted actions depending on the environment"""
        action = ''
        for i in range(self.environment.action_length):
            action += str(random.randint(0, 1))
        return action
