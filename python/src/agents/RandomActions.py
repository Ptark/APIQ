from typing import Tuple, Type
import random

from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


class RandomActions(Agent):
    """Models an agent which takes handcrafted actions depending on environments"""

    def __init__(self, environment_class: Type[Environment]):
        super().__init__(environment_class)

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Returns handcrafted actions depending on the environment"""
        action = ''
        for i in range(self.environment_class.action_length):
            action += str(random.randint(0, 1))
        return action
