import dis
import time
import math
from pathlib import Path
from typing import Callable

from python.src.environments.abstract_classes.Environment import Environment

randomness_complexity = 25


def get_scaling_factor(environment: Environment) -> float:
    """Calculate the scaling factor from the complexity of an environment"""
    return pow(2, -environment_complexity(environment) / randomness_complexity)


def environment_complexity(environment: Environment) -> float:
    """Estimate complexity of an environment depending on its calculate_action method and use of random.
    Complexity is measured in bytecode instructions.
    One instruction has 2 byte.
    """
    complexity = method_complexity(environment.calculate_percept)
    if environment.has_randomness:
        complexity += randomness_complexity
    return complexity


def method_complexity(method: Callable) -> float:
    """Estimate complexity of a method by counting bytecode instructions"""
    bytecode = dis.Bytecode(method).dis()
    return len([line for line in bytecode.splitlines() if line])


def get_reward_from_bitstring(s: str) -> float:
    """Calculate reward from bit string"""
    reward = 0
    sign = 1 if s[0] == "0" else -1
    for idx in range(len(s) - 1):
        reward += int(s[idx + 1]) * pow(0.5, idx)
    return sign * reward


def get_random_bit() -> int:
    """Returns a pseudorandom bit from a timestamp"""
    seed = time.time()
    return pow(2, int(str(seed).replace('.', '')[-5:])) % 3 % 2


def get_data_path() -> Path:
    """Returns resources path."""
    return Path(__file__).parent.parent.joinpath('resources/data')


def is_saved(training_step: int) -> bool:
    """Returns boolean indicating if the given training step is to be saved or loaded."""
    if training_step == 0:
        return False
    if math.log10(training_step).is_integer():
        return True
    if math.log10(training_step * 2).is_integer():
        return True
    return False


def nested_set(dic, keys, value):
    """Sets a nested value in a dictionary"""
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

