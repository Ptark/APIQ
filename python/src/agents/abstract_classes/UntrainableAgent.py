from abc import abstractmethod
from pathlib import Path
from typing import Type

from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


class UntrainableAgent(Agent):
    """Abstract class models an untrainable agent"""

    @abstractmethod
    def __init__(self, environment_class: Type[Environment], path: Path = ''):
        super().__init__(environment_class)

