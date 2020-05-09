from abc import abstractmethod
from pathlib import Path
from typing import Type

from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


class TrainableAgent(Agent):
    """Abstract class models a trainable agent"""

    @abstractmethod
    def __init__(self, environment_class: Type[Environment], path: Path = ''):
        super().__init__(environment_class)

    @abstractmethod
    def train(self, label: str):
        """To be implemented by child classes
        trains an agent in the given environment with the given reward sign_bit"""
        pass

    @abstractmethod
    def save(self, path: Path):
        """To be implemented by child classes
        Saves learning to file"""
        pass
