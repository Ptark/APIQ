from abc import ABC, abstractmethod


class Environment(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calculate_percept(self, action):
        pass
