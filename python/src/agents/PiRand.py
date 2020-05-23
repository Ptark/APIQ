from python.src import Utility
from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


class PiRand(Agent):
    """Models an agent which takes random actions."""

    def __init__(self, environment: Environment):
        super().__init__(environment)

    def calculate_action(self, observation: str) -> str:
        """Returns a random action."""
        return Utility.random_action(self.environment.action_length)
