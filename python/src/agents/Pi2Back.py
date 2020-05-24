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


class Pi2Back(Agent):
    """The agent keeping a table with (observation action reward)^2 statistics"""

    def __init__(self, environment: Environment):
        super().__init__(environment)
        self.table = self.init_table()
        # last_action
        self.l_a = "0" * self.environment.action_length
        # last_observation
        self.l_o = "0" * self.environment.observation_length
        # observation
        self.o = "0" * self.environment.observation_length
        self.action_statistic = (1, '', 0)

    def calculate_action(self, observation: str) -> str:
        """ 10% of the time return random action.
            90% of the time return action with most expected reward for
            h = last_observation last_action observation"""
        self.l_o = self.o
        self.o = observation
        if random.randint(0, 9) == 0:
            idx = random.randint(0, len(self.table[self.l_o][self.l_a][self.o]) - 1)
            self.action_statistic = self.table[self.l_o][self.l_a][self.o].pop(idx)
            heapq.heapify(self.table[self.l_o][self.l_a][self.o])
            return self.action_statistic[1]
        else:
            self.action_statistic = heapq.heappop(self.table[self.l_o][self.l_a][self.o])
            return self.action_statistic[1]

    def train(self, reward: str):
        """Add returned reward to statistic"""
        reward_value = -1 * Utility.get_reward_from_bitstring(reward)
        expected_reward = self.action_statistic[0]
        action = self.action_statistic[1]
        cnt = self.action_statistic[2]
        new_expected_reward = (expected_reward * cnt + reward_value) / (cnt + 1)
        heapq.heappush(self.table[self.l_o, self.l_a, self.o], (new_expected_reward, action, cnt + 1))

    def init_table(self) -> dict:
        """Returns a dict where every observation has an ordered list of actions with rewards"""
        table = {}
        a_length = self.environment.action_length
        o_length = self.environment.observation_length
        if o_length == 0:
            table[''] = {}
            for action_idx in range(pow(2, a_length)):
                action = Utility.get_bitstring_from_decimal(action_idx, a_length)
                Utility.nested_set(table, ['', action, ''], init_action_heap(a_length))
        for last_observation_idx in range(pow(2, o_length)):
            last_observation = Utility.get_bitstring_from_decimal(last_observation_idx, o_length)
            for action_idx in range(pow(2, a_length)):
                action = Utility.get_bitstring_from_decimal(action_idx, a_length)
                for observation_idx in range(pow(2, o_length)):
                    observation = Utility.get_bitstring_from_decimal(observation_idx, o_length)
                    Utility.nested_set(table, [last_observation, action, observation], init_action_heap(a_length))
        return table


