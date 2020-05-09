from typing import Tuple, Type

from python.src.agents.abstract_classes.UntrainableAgent import UntrainableAgent
from python.src.environments.abstract_classes.Environment import Environment


class Handcrafted(UntrainableAgent):
    """Models an agent which takes handcrafted actions depending on environments"""

    def __init__(self, environment_class: Type[Environment]):
        """Initiate switch dictionaries for choosing environment appropriate actions"""
        super().__init__(environment_class)
        self.environment_switch = {
            "BiasedCoinFlip": self.biased_coin_flip,
            "BiasedCoinFlipR": self.biased_coin_flip_reversed,
            "DoubleCoinFlip": self.double_coin_flip,
            "DoubleCoinFlipR": self.double_coin_flip_reversed,
            "Slide": self.slide,
            "SlideR": self.slide_reversed,
            "AnyOne": self.any_one,
            "AnyOneR": self.any_one_reversed,
            "SpecificOne": self.specific_one,
            "SpecificOneR": self.specific_one_reversed,
        }
        self.turn_counter = 0

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Returns handcrafted actions depending on the environment"""
        getter = self.environment_switch.get
        action = getter(self.environment_class.__name__)(percept)
        self.turn_counter += 1
        return action

    def biased_coin_flip(self, percept: Tuple[str, str]):
        """Returns optimal actions for the biased coin flip"""
        return "1"

    def biased_coin_flip_reversed(self, percept: Tuple[str, str]):
        """Returns optimal actions for the reversed biased coin flip"""
        return "0"

    def double_coin_flip(self, percept: Tuple[str, str]):
        """Returns optimal actions for the double coin flip"""
        return "01"

    def double_coin_flip_reversed(self, percept: Tuple[str, str]):
        """Returns optimal actions for the double coin flip"""
        return "00"

    def slide(self, percept: Tuple[str, str]):
        """Returns optimal actions for the Slide"""
        if self.turn_counter == 0:
            return "1"
        else:
            return "1"

    def slide_reversed(self, percept: Tuple[str, str]):
        """Returns optimal actions for the Slide"""
        if self.turn_counter == 0:
            return "1"
        else:
            return "0"

    def any_one(self, percept: Tuple[str, str]):
        """Returns optimal action for AnyOne"""
        return "11"

    def any_one_reversed(self, percept: Tuple[str, str]):
        """Returns optimal action for reversed AnyOne"""
        return "00"

    def specific_one(self, percept: Tuple[str, str]):
        """Returns optimal action for SpecificOne"""
        return "11"

    def specific_one_reversed(self, percept: Tuple[str, str]):
        """Returns optimal action for reversed SpecificOne"""
        return "00"
