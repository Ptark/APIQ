import dis
import time
from typing import List, Callable

from python.src.agents.Agent import Agent
from python.src.agents.HandcraftedAgent import HandcraftedAgent
from python.src.agents.RandomAgent import RandomAgent
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip
from python.src.environments.DoubleCoinFlip import DoubleCoinFlip
from python.src.environments.Environment import Environment
from python.src.environments.Slide import Slide

floating_precision_factor = 3400
randomness_complexity = 25


def get_scaling_factor(environment: Environment, randomness: bool = False) -> float:
    """Calculate the scaling factor from the complexity of an environment"""
    return 1 / pow(2, environment_complexity(environment, randomness) / floating_precision_factor)


def environment_complexity(environment: Environment, randomness: bool = False) -> float:
    """Estimate complexity of an environment depending on its main method and use of random"""
    complexity = method_complexity(environment.calculate_percept)
    if randomness:
        complexity += randomness_complexity
    return complexity


def method_complexity(method: Callable) -> float:
    """Estimate complexity of a method by counting bytecode instructions"""
    # bits per instruction = 16
    bytecode = dis.Bytecode(method).dis()
    return len([line for line in bytecode.splitlines() if line])


def get_random_bit() -> int:
    """Returns a pseudorandom bit from a timestamp"""
    seed = time.time()
    return pow(2, int(str(seed).replace('.', '')[-5:])) % 3 % 2


def agents() -> List[type(Agent)]:
    """Initialize and return a list of agents"""
    return [
        RandomAgent,
        HandcraftedAgent
    ]


def environments() -> List[type(Environment)]:
    """Initialize and return a list of environments"""
    return [
        BiasedCoinFlip,
        DoubleCoinFlip,
        Slide,
    ]
