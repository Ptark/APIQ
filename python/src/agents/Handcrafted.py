from typing import Tuple

from python.src.agents.Agent import Agent


class Handcrafted(Agent):
    """Models an agent which takes handcrafted actions depending on environments"""

    training_steps = 1

    def __init__(self, training_step: int = 0):
        """Initiate switch dictionaries for choosing environment appropriate actions"""
        super().__init__(training_step)
        self.environment_switch = {
            0: self.biased_coin_flip,
            1: self.double_coin_flip,
            2: self.slide
        }
        self.environment_reversed_switch = {
            0: self.biased_coin_flip_reversed,
            1: self.double_coin_flip_reversed,
            2: self.slide_reversed
        }
        self.environment_solver = None

    def calculate_action(self, percept: Tuple[str, str]) -> str:
        """Returns handcrafted actions depending on the environment"""
        if self.turn_counter == 0:
            self.idx = int(percept[0])
            self.sign = 1 - 2 * (int(percept[1]) == 1)
            if self.sign == 1:
                self.environment_solver = self.environment_switch.get
            else:
                self.environment_solver = self.environment_reversed_switch.get
        action = self.environment_solver(self.idx)(percept)
        self.turn_counter += 1
        return action

    def biased_coin_flip(self, percept: Tuple[str, str]):
        """Returns optimal actions for the biased coin flip"""
        return "0001"

    def biased_coin_flip_reversed(self, percept: Tuple[str, str]):
        """Returns optimal actions for the reversed biased coin flip"""
        return "0000"

    def double_coin_flip(self, percept: Tuple[str, str]):
        """Returns optimal actions for the double coin flip"""
        return "0001"

    def double_coin_flip_reversed(self, percept: Tuple[str, str]):
        """Returns optimal actions for the double coin flip"""
        return "0000"

    def slide(self, percept: Tuple[str, str]):
        """Returns optimal actions for the slide"""
        if self.turn_counter == 0:
            return "0001"
        else:
            return "0001"

    def slide_reversed(self, percept: Tuple[str, str]):
        """Returns optimal actions for the slide"""
        if self.turn_counter == 0:
            return "0001"
        else:
            return "0000"