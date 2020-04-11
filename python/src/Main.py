import dis
import pprint

from python.src import Utility
from python.src.environments import APIQ
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip


def main():
    """Calculate apiq with number_of_evaluations"""
    number_of_evaluations = 20000  # number of evaluations of an agent in an environment for stochastic purposes
    pprint.pprint(APIQ.complexity(), sort_dicts=False)
    pprint.pprint(APIQ.apiq(number_of_evaluations), sort_dicts=False)


if __name__ == '__main__':
    main()


