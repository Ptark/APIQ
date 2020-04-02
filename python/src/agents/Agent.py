from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calculate_action(self, percept):
        pass
