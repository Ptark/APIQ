import pprint

from python.src import APIQ


def main():
    """Calculate apiq with number_of_evaluations"""
    APIQ.calculate_scaling_factors()
    pprint.pprint(APIQ.environment_scaling_factors)
    APIQ.train()
    APIQ.calculate_rewards()
    APIQ.calculate_apiq()
    for agent in APIQ.apiq_dict:
        print(agent)
        print(APIQ.apiq_dict[agent]["apiq"])


if __name__ == '__main__':
    main()


