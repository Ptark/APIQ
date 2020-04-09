from typing import Tuple
import random

from python.src.agents.Agent import Agent


class HandcraftedAgent(Agent):
    """Models an agent which takes handcrafted actions depending on environments"""

    def __init__(self):
        super().__init__()

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Returns handcrafted actions depending on the environment"""
        action = ''
        for i in range(4):
            action += str(random.randint(0, 1))
        return action
