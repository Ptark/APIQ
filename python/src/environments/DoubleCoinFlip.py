from random import random


class DoubleCoinFlip:
    """Class models a double coin flip. Returns 1, if the """

    def __init__(self):
        pass

    def flip(self, prediction):
        outcome = random.randint(0, 1) + random.randint(0, 1)
        if prediction == outcome:
            return 1
        else:
            return 0
