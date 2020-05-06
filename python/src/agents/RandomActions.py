from typing import Tuple, Type
import random

from python.src.agents.Agent import Agent
from python.src.environments.Environment import Environment


class RandomActions(Agent):
    """Models an agent which takes handcrafted actions depending on environments"""

    has_randomness = True

    def __init__(self, environment_class: Type[Environment], sign_bit: str = "0", training_step: int = 0):
        super().__init__(environment_class, sign_bit, training_step)

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Returns handcrafted actions depending on the environment"""
        action = ''
        for i in range(self.environment_class.action_length):
            action += str(random.randint(0, 1))
        return action
