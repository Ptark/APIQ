import heapq
import random
from python.src import Utility
from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


class PiBasic(Agent):
    """The basic agent keeping a table with observation - action - reward statistics"""

    def __init__(self, environment: Environment):
        super().__init__(environment)
        self.table = self.init_table()
        self.observation = "0" * self.environment.observation_length
        self.action_statistic = (0, "0" * self.environment.action_length, 0)

    def calculate_action(self, observation: str) -> str:
        """ 10% of the time return random action.
            90% of the time return action with most expected reward for observation"""
        self.observation = observation
        if random.randrange(0, 10) == 0:
            idx = random.randrange(0, len(self.table[observation]))
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
        heapq.heappush(self.table[self.observation], (new_expected_reward, action, cnt + 1))

    def init_table(self) -> dict:
        """Returns a dict where every observation has an ordered list of actions with rewards"""
        table = {}
        a_length = self.environment.action_length
        o_length = self.environment.observation_length
        if o_length == 0:
            table[''] = Utility.init_heap(a_length)
        else:
            for observation_idx in range(pow(2, o_length)):
                observation = format(observation_idx, 'b').zfill(o_length)
                table[observation] = Utility.init_heap(a_length)
        return table


