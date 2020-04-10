from typing import Tuple

from python.src.agents.Agent import Agent


class HandcraftedAgent(Agent):
    """Models an agent which takes handcrafted actions depending on environments"""

    def __init__(self):
        self.environment_switch = {
            "0": self.biased_coin_flip,
            "1": self.double_coin_flip,
            "2": self.slide
        }
        self.environment_reversed_switch = {
            "0": self.biased_coin_flip_reversed,
            "1": self.double_coin_flip_reversed,
            "2": self.slide_reversed
        }
        super().__init__()

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Returns handcrafted actions depending on the environment"""
        if self.turn_counter == 0:
            self.idx = percept[0]
            self.sign = percept[1]
        if self.sign == "0":
            action = self.environment_switch.get(self.idx)(percept)
        else:
            action = self.environment_reversed_switch.get(self.idx)(percept)
        self.turn_counter += 1
        return action

    def biased_coin_flip(self, percept):
        """Returns optimal actions for the biased coin flip"""
        return "1"

    def biased_coin_flip_reversed(self, percept):
        """Returns optimal actions for the reversed biased coin flip"""
        return "0"

    def double_coin_flip(self, percept):
        """Returns optimal actions for the double coin flip"""
        return "1"

    def double_coin_flip_reversed(self, percept):
        """Returns optimal actions for the double coin flip"""
        return "0"

    def slide(self, percept):
        """Returns optimal actions for the slide"""
        if self.turn_counter == 0:
            return "1"
        else:
            return "1"

    def slide_reversed(self, percept):
        """Returns optimal actions for the slide"""
        if self.turn_counter == 0:
            return "1"
        else:
            return "0"
