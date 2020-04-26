from typing import Tuple

from python.src.agents.Agent import Agent


class NNAgentTemplate(Agent):
    """Template for neural network based agents"""

    def __init__(self):
        """Load appropriate parameters depending on environment and learning time"""
        super().__init__()

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Calculate action based on neural network"""

