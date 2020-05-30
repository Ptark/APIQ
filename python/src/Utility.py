import dis
import heapq
import time
import math
from pathlib import Path
import random
from typing import Callable

from python.src.environments.abstract_classes.Environment import Environment

randomness_complexity = 25


def calculate_complexity(environment: Environment) -> float:
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


def get_decimal_from_bitstring(s: str) -> float:
    """Calculate decimal number from bitstring"""
    decimal = 0
    for idx in range((len(s) - 1)):
        decimal += int(s[len(s) - 1 - idx]) * pow(2, idx)
    return decimal


def get_bitstring_from_decimal(decimal: int, length: int) -> str:
    """Calculate bitstring from decimal"""
    return format(decimal, 'b').zfill(length)


def get_random_bit() -> int:
    """Returns a pseudorandom bit from a timestamp. Used for calculating random bit complexity"""
    seed = time.time()
    return pow(2, int(str(seed).replace('.', '')[-5:])) % 3 % 2


def get_data_path() -> Path:
    """Returns data path."""
    return Path(__file__).parent.parent.joinpath('resources/data')


def get_plots_path() -> Path:
    """Returns plots path."""
    return Path(__file__).parent.parent.joinpath('resources/plots')


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


def random_action(length: int) -> str:
    """Returns a random action"""
    action = ''
    for i in range(length):
        action += str(random.randint(0, 1))
    return action


def init_heap(length: int):
    """initialize action heap for pi agents"""
    reward_statistics = []
    for action_idx in range(pow(2, length)):
        action = get_bitstring_from_decimal(action_idx, length)
        heapq.heappush(reward_statistics, (1, action, 0))
    return reward_statistics


def heapq_siftdown(heap, startpos, pos):
    """Taken from heapq internal code since it might be deprecated
    https://hg.python.org/cpython/file/3.6/Lib/heapq.py
    Implements decrease_key"""
    newitem = heap[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


def heapq_siftup(heap, pos):
    """Taken from heapq internal code since it might be deprecated.
    https://hg.python.org/cpython/file/3.6/Lib/heapq.py
    Implements increase_key"""
    endpos = len(heap)
    startpos = pos
    newitem = heap[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2*pos + 1    # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        # Move the smaller child up.
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos] = newitem
    heapq_siftdown(heap, startpos, pos)



