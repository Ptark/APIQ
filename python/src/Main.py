import pprint

from python.src.environments import APIQ


def main():
    number_of_evaluations = 100000  # number of evaluations of an agent in an environment for stochastic purposes
    scores = APIQ.calculate_apiq_scores(number_of_evaluations)
    pprint.pprint(scores, sort_dicts=False)


if __name__ == '__main__':
    main()


