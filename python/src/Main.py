import pprint

from python.src import APIQ


number_of_evaluations = 1000  # number of evaluations of an agent in an environment for stochastic purposes
training_steps = 1001
step_size = 200


def main():
    """Calculate apiq with number_of_evaluations"""
    APIQ.train()
    pprint.pprint(APIQ.complexity(), sort_dicts=False)
    pprint.pprint(APIQ.apiq(), sort_dicts=False)


if __name__ == '__main__':
    main()


