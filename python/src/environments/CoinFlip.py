from random import random
from python.src.environments.Environment import Environment


class CoinFlip(Environment):
    def __init__(self):
        super().__init__()

    def calculate_percept(self, prediction):
        outcome = random.randint(0, 1)
        if prediction == outcome:
            return "0001", "1000"
        else:
            return "0000", "0000"
