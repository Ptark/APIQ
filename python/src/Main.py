from python.src import Utility
from python.src.environments import APIQ


def main():
    number_of_evaluations = 10000  # number of evaluations of an agent in an environment for stochastic purposes
    scores = APIQ.calculate_apiq_scores(number_of_evaluations)
    print(scores)


if __name__ == '__main__':
    main()


