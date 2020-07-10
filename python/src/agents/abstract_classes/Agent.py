import random
from abc import ABC, abstractmethod
from python.src.environments.abstract_classes.Environment import Environment


class Agent(ABC):
    """Abstract class models an agent"""

    @abstractmethod
    def __init__(self, environment: Environment, seed: int = 1):
        self.environment = environment
        self.seed = seed

    @abstractmethod
    def calculate_action(self, observation: str) -> str:
        """To be implemented by child classes
        returns an action as a string of 0s and 1s"""
        pass

    def train(self, reward: str):
        """Trains agent on observation and reward if trainable"""
        pass

    def seeded_rand_range(self, low: int = 0, high: int = 2):
        """Returns a seeded randrange and increments internal seed"""
        random.seed(self.seed)
        self.seed += 1
        return random.randrange(low, high)
