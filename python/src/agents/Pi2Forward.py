import heapq
from python.src import Utility
from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment


class Pi2Forward(Agent):
    """The agent keeping a table with statistics depending on the last observation
    and the next possible actions"""

    def __init__(self, environment: Environment):
        super().__init__(environment)
        self.cnt = 0
        a_length = self.environment.action_length
        o_length = self.environment.observation_length
        self.table = self.init_table()
        # second_to_last_observation
        self.sl_o = "0" * o_length if o_length > 0 else ''
        # second_to_last_action
        self.sl_a = "0" * a_length
        # last_observation
        self.l_o = "0" * o_length if o_length > 0 else ''
        # last_action
        self.l_a = "0" * a_length
        # observation
        self.o = "0" * o_length if o_length > 0 else ''
        # reward
        self.r = 0
        # action
        self.a = "0" * a_length
        # new_reward
        self.nr = 0

    def calculate_action(self, observation: str) -> str:
        """ 10% of the time return random action.
            90% of the time return action with most expected reward for the next
            2 actions and h = last_observation last_action observation"""
        self.sl_o = self.l_o
        self.l_o = self.o
        self.o = observation
        self.sl_a = self.l_a
        self.l_a = self.a
        action_heap = self.table[self.l_o][self.l_a][self.o]
        if self.seeded_rand_range(10) == 0:
            idx = self.seeded_rand_range(self.table[self.l_o][self.l_a][self.o])
            self.a = action_heap[idx][1][:self.environment.action_length]
        else:
            self.a = action_heap[0][1][:self.environment.action_length]
        return self.a

    def train(self, reward: str):
        """Add returned reward to statistic"""
        self.cnt += 1
        self.r = self.nr
        self.nr = -1 * Utility.get_reward_from_bitstring(reward)
        action_heap = self.table[self.sl_o][self.sl_a][self.l_o]
        for idx in range(len(action_heap)):
            if action_heap[idx][1] == self.l_a + self.a:
                expected_reward = action_heap[idx][0]
                reward = self.r + self.nr
                cnt = action_heap[idx][2]
                new_expected_reward = (expected_reward * cnt + reward) / (cnt + 1)
                action_heap[idx] = (new_expected_reward, action_heap[idx][1], cnt + 1)
                if expected_reward < reward:
                    Utility.heapq_siftup(action_heap, idx)
                else:
                    Utility.heapq_siftdown(action_heap, 0, idx)
                break

    def init_table(self) -> dict:
        """Returns a dict with reward statistics for every combination of oao|arar"""
        table = {}
        a_length = self.environment.action_length
        o_length = self.environment.observation_length
        if o_length == 0:
            table[''] = {}
            for action_idx in range(pow(2, a_length)):
                action = Utility.get_bitstring_from_decimal(action_idx, a_length)
                table[''][action] = {}
                table[''][action][''] = init_double_heap(a_length)
        else:
            for last_observation_idx in range(pow(2, o_length)):
                last_observation = Utility.get_bitstring_from_decimal(last_observation_idx, o_length)
                table[last_observation] = {}
                for action_idx in range(pow(2, a_length)):
                    action = Utility.get_bitstring_from_decimal(action_idx, a_length)
                    table[last_observation][action] = {}
                    for observation_idx in range(pow(2, o_length)):
                        observation = Utility.get_bitstring_from_decimal(observation_idx, o_length)
                        table[last_observation][action][observation] = init_double_heap(a_length)
        return table


def init_double_heap(length: int):
    """initialize heap with 2 actions for pi2forward agent"""
    reward_statistics = []
    for action_one_idx in range(pow(2, length)):
        action_one = Utility.get_bitstring_from_decimal(action_one_idx, length)
        for action_two_idx in range(pow(2, length)):
            action_two = Utility.get_bitstring_from_decimal(action_two_idx, length)
            # (expected_reward, actions, number_of_times)
            heapq.heappush(reward_statistics, (2, action_one + action_two, 0))
    return reward_statistics
