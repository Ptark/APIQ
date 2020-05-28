from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


class Handcrafted(Agent):
    """Models an agent which takes handcrafted actions depending on environments"""

    def __init__(self, environment: Environment):
        """Initiate switch dictionaries for choosing environment appropriate actions"""
        super().__init__(environment)
        self.environment_switch = {
            "BiasedCoinFlip0": (self.biased_coin_flip, self.trainer_pass),
            "BiasedCoinFlip1": (self.biased_coin_flip_reversed, self.trainer_pass),
            "DoubleCoinFlip0": (self.double_coin_flip, self.trainer_pass),
            "DoubleCoinFlip1": (self.double_coin_flip_reversed, self.trainer_pass),
            "Slide0": (self.slide, self.trainer_pass),
            "Slide1": (self.slide_reversed, self.trainer_pass),
            "SlideTwo0": (self.slide_two, self.trainer_pass),
            "SlideTwo1": (self.slide_two_reversed, self.trainer_pass),
            "AnyOne0": (self.any_one, self.trainer_pass),
            "AnyOne1": (self.any_one_reversed, self.trainer_pass),
            "SpecificOne0": (self.specific_one, self.trainer_pass),
            "SpecificOne1": (self.specific_one_reversed, self.trainer_pass),
            "Switch0": (self.switch, self.trainer_pass),
            "Switch1": (self.switch_reversed, self.trainer_pass),
            "Alternate0": (self.alternate, self.trainer_pass),
            "Alternate1": (self.alternate_reversed, self.trainer_pass),
            "Button0": (self.button, self.trainer_pass),
            "Button1": (self.button_reversed, self.trainer_pass),
            "CombinationTwo0": (self.combination_two, self.trainer_pass),
            "CombinationTwo1": (self.combination_two_reversed, self.trainer_pass),
            "CombinationThree0": (self.combination_three, self.trainer_pass),
            "CombinationThree1": (self.combination_three_reversed, self.trainer_pass),
            "CombinationFour0": (self.combination_four, self.trainer_pass),
            "CombinationFour1": (self.combination_four_reversed, self.trainer_pass),
            "AlternateRandomly0": (self.alternate_randomly, self.trainer_alternate_randomly),
            "AlternateRandomly1": (self.alternate_randomly_reversed, self.trainer_alternate_randomly_reversed),
            "Labyrinth0": (self.labyrinth, self.trainer_pass),
            "Labyrinth1": (self.labyrinth_reversed, self.trainer_pass),
            "LabyrinthCoord0": (self.labyrinth, self.trainer_pass),
            "LabyrinthCoord1": (self.labyrinth_reversed, self.trainer_pass),
            "LabyrinthLoop0": (self.labyrinth_loop, self.trainer_pass),
            "LabyrinthLoop1": (self.labyrinth_loop_reversed, self.trainer_pass),
        }
        self.sign_bit = self.environment.sign_bit
        self.turn_counter = 0
        self.boolean = True
        methods = self.environment_switch.get(self.environment.__class__.__name__ + self.sign_bit)
        self.calculator = methods[0]
        self.trainer = methods[1]
        if self.calculator is None:
            print("!!!")
            print("Handcrafted needs implementation for %s" % self.environment.__class__.__name__)
            print("Implement it and delete the relevant save files in python/resources/data")

    def calculate_action(self, observation: str) -> str:
        """Returns handcrafted actions depending on the environment"""
        try:
            action = self.calculator(observation)
        except TypeError:
            action = "0" * self.environment.action_length
        self.turn_counter += 1
        return action

    def train(self, reward: str):
        """Trains the environment for optimal actions"""
        try:
            self.trainer(reward)
        except TypeError:
            pass

    def trainer_pass(self, reward: str):
        """Trainer dummy if no training is necessary"""
        pass

    def biased_coin_flip(self, observation: str) -> str:
        """Returns optimal actions for the biased coin flip"""
        return "1"

    def biased_coin_flip_reversed(self, observation: str) -> str:
        """Returns optimal actions for the reversed biased coin flip"""
        return "0"

    def double_coin_flip(self, observation: str) -> str:
        """Returns optimal actions for the double coin flip"""
        return "01"

    def double_coin_flip_reversed(self, observation: str) -> str:
        """Returns optimal actions for the double coin flip"""
        return "00"

    def slide(self, observation: str) -> str:
        """Returns optimal actions for the Slide"""
        return "1"

    def slide_reversed(self, observation: str) -> str:
        """Returns optimal actions for the Slide"""
        return "0"

    def slide_two(self, observation: str) -> str:
        """Returns optimal actions for the SlideTwo"""
        return "1"

    def slide_two_reversed(self, observation: str) -> str:
        """Returns optimal actions for SlideTwoReversed"""
        return "0"

    def any_one(self, observation: str) -> str:
        """Returns optimal action for AnyOne"""
        return "11"

    def any_one_reversed(self, observation: str) -> str:
        """Returns optimal action for reversed AnyOne"""
        return "00"

    def specific_one(self, observation: str) -> str:
        """Returns optimal action for SpecificOne"""
        return "11"

    def specific_one_reversed(self, observation: str) -> str:
        """Returns optimal action for reversed SpecificOne"""
        return "00"

    def switch(self, observation: str) -> str:
        """Returns optimal actions for switch"""
        return "0" if self.turn_counter % 2 == 0 else "1"

    def switch_reversed(self, observation: str) -> str:
        """Returns optimal actions for switch reversed"""
        return "1" if self.turn_counter % 2 == 0 else "0"

    def alternate(self, observation: str) -> str:
        """Returns optimal actions for alternate"""
        if self.boolean:
            self.boolean = False
            return "0"
        else:
            self.boolean = True
            return "1"

    def alternate_reversed(self, observation: str) -> str:
        """Returns optimal actions for alternate reversed"""
        return "1"

    def button(self, observation: str) -> str:
        """Returns optimal actions for button"""
        return "1"

    def button_reversed(self, observation: str) -> str:
        """Returns optimal actions for button"""
        return "0"

    def combination_two(self, observation: str) -> str:
        """Returns optimal actions for combination_two"""
        if self.boolean:
            self.boolean = False
            return "01"
        else:
            self.boolean = True
            return "11"

    def combination_two_reversed(self, observation: str) -> str:
        """Returns optimal actions for combination_two_reversed"""
        return "00"

    def combination_three(self, observation: str) -> str:
        """Returns optimal actions for combination_three"""
        if self.turn_counter % 3 == 0:
            return "10"
        elif (self.turn_counter + 2) % 3 == 0:
            return "00"
        return "11"

    def combination_three_reversed(self, observation: str) -> str:
        """Returns optimal actions for combination_three_reversed"""
        return "00"

    def combination_four(self, observation: str) -> str:
        """Returns optimal actions for combination_four"""
        if self.turn_counter % 4 == 0:
            return "01"
        elif (self.turn_counter + 3) % 4 == 0:
            return "00"
        elif (self.turn_counter + 2) % 4 == 0:
            return "00"
        return "11"

    def combination_four_reversed(self, observation: str) -> str:
        """Returns optimal actions for combination_three_reversed"""
        return "00"

    def alternate_randomly(self, observation: str) -> str:
        """Returns optimal actions for alternate_randomly"""
        if self.boolean:
            return "0"
        return "1"

    def trainer_alternate_randomly(self, reward: str):
        """Trains agent to act optimally for alternate_randomly"""
        if reward == "00":
            self.boolean = not self.boolean

    def alternate_randomly_reversed(self, observation: str) -> str:
        """Returns optimal actions for alternate_randomly_reversed"""
        if self.boolean:
            return "1"
        return "0"

    def trainer_alternate_randomly_reversed(self, reward: str):
        """Trains agent to act optimally for alternate_randomly_reversed"""
        if reward == "11":
            self.boolean = not self.boolean

    def labyrinth(self, observation: str) -> str:
        """Returns optimal actions for labyrinth"""
        self.boolean = not self.boolean
        if self.boolean:
            return "10"
        else:
            return "11"

    def labyrinth_reversed(self, observation: str) -> str:
        """Returns optimal actions for labyrinth_reversed"""
        return "00"

    def labyrinth_coord(self, observation: str) -> str:
        """Returns optimal actions for labyrinth_coord"""
        self.boolean = not self.boolean
        if self.boolean:
            return "10"
        else:
            return "11"

    def labyrinth_coord_reversed(self, observation: str) -> str:
        """Returns optimal actions for labyrinth_coord_reversed"""
        return "00"

    def labyrinth_loop(self, observation: str) -> str:
        """Returns optimal actions for labyrinth_loop"""
        self.boolean = not self.boolean
        if self.boolean:
            return "00"
        else:
            return "01"

    def labyrinth_loop_reversed(self, observation: str) -> str:
        """Returns optimal actions for labyrinth_loop_reversed"""
        return "00"
