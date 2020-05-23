from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


class Handcrafted(Agent):
    """Models an agent which takes handcrafted actions depending on environments"""

    def __init__(self, environment: Environment):
        """Initiate switch dictionaries for choosing environment appropriate actions"""
        super().__init__(environment)
        self.environment_switch = {
            "BiasedCoinFlip0": self.biased_coin_flip,
            "BiasedCoinFlip1": self.biased_coin_flip_reversed,
            "DoubleCoinFlip0": self.double_coin_flip,
            "DoubleCoinFlip1": self.double_coin_flip_reversed,
            "Slide0": self.slide,
            "Slide1": self.slide_reversed,
            "AnyOne0": self.any_one,
            "AnyOne1": self.any_one_reversed,
            "SpecificOne0": self.specific_one,
            "SpecificOne1": self.specific_one_reversed,
            "Alternate0": self.alternate,
            "Alternate1": self.alternate_reversed,
            "AlternateHidden0": self.alternate_hidden,
            "AlternateHidden1": self.alternate_hidden_reversed,
        }
        self.sign_bit = self.environment.sign_bit
        self.turn_counter = 0

    def calculate_action(self, observation: str) -> str:
        """Returns handcrafted actions depending on the environment"""
        getter = self.environment_switch.get
        action = getter(self.environment.__class__.__name__ + self.sign_bit)(observation)
        self.turn_counter += 1
        return action

    def biased_coin_flip(self, observation: str):
        """Returns optimal actions for the biased coin flip"""
        return "1"

    def biased_coin_flip_reversed(self, observation: str):
        """Returns optimal actions for the reversed biased coin flip"""
        return "0"

    def double_coin_flip(self, observation: str):
        """Returns optimal actions for the double coin flip"""
        return "01"

    def double_coin_flip_reversed(self, observation: str):
        """Returns optimal actions for the double coin flip"""
        return "00"

    def slide(self, observation: str):
        """Returns optimal actions for the Slide"""
        return "1"

    def slide_reversed(self, observation: str):
        """Returns optimal actions for the Slide"""
        if self.turn_counter == 0:
            return "1"
        else:
            return "0"

    def any_one(self, observation: str):
        """Returns optimal action for AnyOne"""
        return "11"

    def any_one_reversed(self, observation: str):
        """Returns optimal action for reversed AnyOne"""
        return "00"

    def specific_one(self, observation: str):
        """Returns optimal action for SpecificOne"""
        return "11"

    def specific_one_reversed(self, observation: str):
        """Returns optimal action for reversed SpecificOne"""
        return "00"

    def alternate(self, observation: str):
        """Returns optimal actions for alternate"""
        return "0" if self.turn_counter % 2 == 0 else "1"

    def alternate_reversed(self, observation: str):
        """Returns optimal actions for alternate"""
        return "1" if self.turn_counter % 2 == 0 else "0"

    def alternate_hidden(self, observation: str):
        """Returns optimal actions for alternate"""
        return "0" if self.turn_counter % 2 == 0 else "1"

    def alternate_hidden_reversed(self, observation: str):
        """Returns optimal actions for alternate"""
        return "1" if self.turn_counter % 2 == 0 else "0"
