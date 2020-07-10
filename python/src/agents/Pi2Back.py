import heapq
from python.src import Utility
from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


class Pi2Back(Agent):
    """The agent keeping a table with reward statistics depending on the
    last observation and action"""

    def __init__(self, environment: Environment, seed: int):
        super().__init__(environment, seed)
        a_length = self.environment.action_length
        o_length = self.environment.observation_length
        self.table = self.init_table()
        # last_action
        self.l_a = "0" * a_length
        # last_observation
        self.l_o = "0" * o_length if o_length > 0 else ''
        # observation
        self.o = "0" * o_length if o_length > 0 else ''
        self.action_statistic = (0, "0" * a_length, 0)

    def calculate_action(self, observation: str) -> str:
        """ 10% of the time return random action.
            90% of the time return action with most expected reward for
            h = last_observation last_action observation"""
        self.l_o = self.o
        self.o = observation
        if self.seeded_rand_range(0, 10) == 0:
            idx = self.seeded_rand_range(0, len(self.table[self.l_o][self.l_a][self.o]))
            self.l_a = self.action_statistic[1]
            self.action_statistic = self.table[self.l_o][self.l_a][self.o].pop(idx)
            heapq.heapify(self.table[self.l_o][self.l_a][self.o])
            return self.action_statistic[1]
        else:
            self.l_a = self.action_statistic[1]
            self.action_statistic = heapq.heappop(self.table[self.l_o][self.l_a][self.o])
            return self.action_statistic[1]

    def train(self, reward: str):
        """Add returned reward to statistic"""
        reward_value = -1 * Utility.get_reward_from_bitstring(reward)
        expected_reward = self.action_statistic[0]
        action = self.action_statistic[1]
        cnt = self.action_statistic[2]
        new_expected_reward = (expected_reward * cnt + reward_value) / (cnt + 1)
        heapq.heappush(self.table[self.l_o][self.l_a][self.o], (new_expected_reward, action, cnt + 1))

    def init_table(self) -> dict:
        """Returns a dict with reward statistics for every combination of oao|ar"""
        table = {}
        a_length = self.environment.action_length
        o_length = self.environment.observation_length
        if o_length == 0:
            table[''] = {}
            for action_idx in range(pow(2, a_length)):
                action = Utility.get_bitstring_from_decimal(action_idx, a_length)
                table[''][action] = {}
                table[''][action][''] = Utility.init_heap(a_length)
        else:
            for last_observation_idx in range(pow(2, o_length)):
                last_observation = Utility.get_bitstring_from_decimal(last_observation_idx, o_length)
                table[last_observation] = {}
                for action_idx in range(pow(2, a_length)):
                    action = Utility.get_bitstring_from_decimal(action_idx, a_length)
                    table[last_observation][action] = {}
                    for observation_idx in range(pow(2, o_length)):
                        observation = Utility.get_bitstring_from_decimal(observation_idx, o_length)
                        table[last_observation][action][observation] = Utility.init_heap(a_length)
        return table


