from random import random
from python.src.environments.Environment import Environment


class BiasedCoinFlip(Environment):
    def __init__(self):
        super().__init__()

    def calculate_percept(self, prediction):
        outcome = random.randint(0, 2)
        if outcome == 2:
            outcome = 1
        if prediction[0] == outcome:
            return 0001, "0100"
        else:
            return "0000", "0000"
