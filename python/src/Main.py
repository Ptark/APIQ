import pprint

from python.src.environments import APIQ


def main():
    """Calculate apiq with number_of_evaluations"""
    number_of_evaluations = 20000  # number of evaluations of an agent in an environment for stochastic purposes
    scores = APIQ.apiq(number_of_evaluations)
    pprint.pprint(scores, sort_dicts=False)


if __name__ == '__main__':
    main()


