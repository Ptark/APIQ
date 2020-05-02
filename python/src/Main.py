import pprint

from python.src import APIQ


number_of_evaluations = 100  # number of evaluations of an agent in an environment for stochastic purposes


def main():
    """Calculate apiq with number_of_evaluations"""
    APIQ.train()
    pprint.pprint(APIQ.complexity(), sort_dicts=False)
    pprint.pprint(APIQ.apiq(number_of_evaluations), sort_dicts=False)


if __name__ == '__main__':
    main()


