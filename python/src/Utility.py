import dis
import time

from python.src.agents.HandcraftedAgent import HandcraftedAgent
from python.src.agents.RandomAgent import RandomAgent
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip
from python.src.environments.DoubleCoinFlip import DoubleCoinFlip
from python.src.environments.Environment import Environment
from python.src.environments.Slide import Slide


def get_scaling_factor(environment: Environment, randomness: bool = False) -> float:
    """Calculate the scaling factor from the complexity of an environment"""
    return 1 / pow(2, environment_complexity(environment, randomness))


def environment_complexity(environment: Environment, randomness: bool = False) -> int:
    """Estimate complexity of an environment depending on its main method and use of random"""
    complexity = method_complexity(environment.calculate_percept)
    if randomness:
        complexity += method_complexity(get_random_bit())
    return complexity


def method_complexity(method) -> int:
    """Estimate complexity of a method by counting bytecode instructions"""
    bytecode = dis.Bytecode(method).dis()
    return len([line for line in bytecode.splitlines() if line]) * 2


def get_random_bit():
    """Returns a pseudorandom bit from a timestamp"""
    seed = time.time()
    return pow(2, int(str(seed).replace('.', ''))) % 3 % 2


def agents():
    """Initialize and return a list of agents"""
    return [
        RandomAgent,
        HandcraftedAgent
    ]


def environments():
    """Initialize and return a list of environments"""
    return [
        BiasedCoinFlip,
        DoubleCoinFlip,
        Slide,
    ]
