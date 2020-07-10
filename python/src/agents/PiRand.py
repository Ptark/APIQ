from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


class PiRand(Agent):
    """Models an agent which takes random actions."""

    def __init__(self, environment: Environment, seed: int):
        super().__init__(environment, seed)

    def calculate_action(self, observation: str) -> str:
        """Returns a random action."""
        action = ''
        for i in range(self.environment.action_length):
            action += str(self.seeded_rand_range())
        return action
