import heapq
import random
from python.src import Utility
from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


def init_action_heap(length: int):
    reward_statistics = []
    for action_idx in range(pow(2, length)):
        action = format(action_idx, 'b').zfill(length)
        heapq.heappush(reward_statistics, (0, action, 0))
    return reward_statistics


class PiBasic(Agent):
    """The basic agent keeping a table with observation - action - reward statistics"""

    def __init__(self, environment: Environment):
        super().__init__(environment)
        self.table = self.init_table()
        self.observation = ''
        self.action_statistic = (1, '', 0)

    def calculate_action(self, observation: str) -> str:
        """Returns handcrafted actions depending on the environment"""
        self.observation = observation
        if random.randint(0, 9) == 0:
            idx = random.randint(0, len(self.table[observation]) - 1)
            self.action_statistic = self.table[observation].pop(idx)
            heapq.heapify(self.table[observation])
            return self.action_statistic[1]
        else:
            self.action_statistic = heapq.heappop(self.table[observation])
            return self.action_statistic[1]

    def train(self, reward: str):
        """Add returned reward to statistic"""
        reward_value = -1 * Utility.get_reward_from_bitstring(reward)
        expected_reward = self.action_statistic[0]
        action = self.action_statistic[1]
        cnt = self.action_statistic[2]
        new_expected_reward = (expected_reward * cnt + reward_value) / (cnt + 1)
        heapq.heappush(self.table[self.observation], (new_expected_reward, action, cnt))

    def init_table(self) -> dict:
        """Returns a dict where every observation has an ordered list of actions with rewards"""
        table = {}
        a_length = self.environment.action_length
        o_length = self.environment.observation_length
        if o_length == 0:
            table[''] = init_action_heap(a_length)
        for observation_idx in range(pow(2, o_length)):
            observation = format(observation_idx, 'b').zfill(o_length)
            table[observation] = init_action_heap(a_length)
        return table


