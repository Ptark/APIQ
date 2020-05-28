from typing import Tuple
from python.src.environments.abstract_classes.Environment import Environment


class LabyrinthCoord(Environment):
    """Class models a labyrinth where the agent has to get from (0,0) to (3,3)
    The field is a bounded 4x4 square.
    The coordinates are returned as observation
    """

    observation_length = 4
    reward_length = 2
    action_length = 2
    max_average_reward_per_cycle = 1/6

    def __init__(self, sign_bit: str = "0"):
        super().__init__(sign_bit)
        self.coord = [0, 0]

    def calculate_percept(self, action: str) -> Tuple[str, str]:
        """Returns reward 1 if the agent reaches (3,3)"""
        if action == "00":
            self.coord[1] -= 1  # down
        if action == "01":
            self.coord[0] -= 1  # left
        if action == "10":
            self.coord[0] += 1  # right
        if action == "11":
            self.coord[1] += 1  # up
        for idx in range(len(self.coord)):
            if self.coord[idx] < 0:
                self.coord[idx] = 0
            if self.coord[idx] > 3:
                self.coord[idx] = 3
        x_coord_bin = format(self.coord[0], 'b'). zfill(2)
        y_coord_bin = format(self.coord[1], 'b'). zfill(2)
        observation = x_coord_bin + y_coord_bin
        if self.coord == [3, 3]:
            self.coord = [0, 0]
            return observation, self.sign_bit + "1"
        return observation, self.sign_bit + "0"

