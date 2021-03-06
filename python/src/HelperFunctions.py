import dis
from typing import Callable, Tuple, Type

from python.src import Utility
from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment

"""File with function definitions for Main.py"""


def pseudo_random_bit(seed: int = 1):
    """return a pseudo random bit from a seed"""
    a = 48271
    m = pow(2, 31) - 1
    return ((a * seed) % m) % 2


def method_complexity(method: Callable) -> float:
    """Estimate complexity of a method by counting bytecode instructions"""
    bytecode = dis.Bytecode(method).dis()
    return len([line for line in bytecode.splitlines() if line])


randomness_complexity = method_complexity(pseudo_random_bit)


def calculate_complexity(environment: Environment) -> float:
    """Estimate complexity of an environment depending on its calculate_action method and use of random.
    Complexity is measured in bytecode instructions.
    One instruction has 2 byte.
    """
    complexity = method_complexity(environment.calculate_percept)
    if environment.has_randomness:
        complexity += randomness_complexity
    return complexity


def trial(pair: Tuple[Type[Agent], Type[Environment]], sign_bit: str, num_trials: int = 100,
          num_cycles: int = 10000, seed: int = 1) -> Tuple[str, str, str, list]:
    """Trial agent in environment"""
    agent_class, environment_class = pair[0], pair[1]
    ag_name, env_name = agent_class.__name__, environment_class.__name__
    rewards = []
    sign = "positive" if sign_bit == "0" else "negative"
    a, m = 16807, (pow(2, 31) - 1)
    for trial_idx in range(num_trials):
        total_reward = 0
        environment = environment_class(sign_bit, seed=seed)
        seed = (seed * a) % m
        agent = agent_class(environment=environment, seed=seed)
        seed = (seed * a) % m
        observation = "0" * environment.observation_length
        for cycle_idx in range(num_cycles):
            action = agent.calculate_action(observation)
            observation, reward = environment.calculate_percept(action)
            total_reward += Utility.get_reward_from_bitstring(reward)
            agent.train(reward)
        total_reward /= num_cycles * environment_class.max_average_reward_per_cycle
        rewards.append(total_reward)
    return ag_name, env_name, sign, rewards
