import pprint

from python.src import APIQ


def main():
    """Calculate apiq with number_of_evaluations"""
    APIQ.train()
    pprint.pprint(APIQ.complexity(), sort_dicts=False)
    pprint.pprint(APIQ.apiq(), sort_dicts=False)


if __name__ == '__main__':
    main()


