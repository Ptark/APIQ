import dis
import time
from pathlib import Path
from typing import List, Callable

from python.src.agents.NNrelu4443 import NNrelu4443
from python.src.agents.NNsigmoid0 import NNsigmoid0
from python.src.agents.NNtanh0 import NNtanh0
from python.src.agents.RandomActions import RandomActions
from python.src.agents.Handcrafted import Handcrafted
from python.src.agents.Agent import Agent
from python.src.agents.NNrelu0 import NNrelu0
from python.src.environments.SpecificOne import SpecificOne
from python.src.environments.AnyOne import AnyOne
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip
from python.src.environments.DoubleCoinFlip import DoubleCoinFlip
from python.src.environments.Environment import Environment
from python.src.environments.Slide import Slide

randomness_complexity = 25


def get_scaling_factor(environment: Environment, randomness: bool = False) -> float:
    """Calculate the scaling factor from the complexity of an environment"""
    return pow(2, -environment_complexity(environment, randomness))


def environment_complexity(environment: Environment, randomness: bool = False) -> float:
    """Estimate complexity of an environment depending on its calculate_action method and use of random.
    Complexity is measured in bytecode instructions.
    One instruction has 2 byte.
    """
    complexity = method_complexity(environment.calculate_percept)
    if randomness:
        complexity += randomness_complexity
    return complexity / randomness_complexity


def method_complexity(method: Callable) -> float:
    """Estimate complexity of a method by counting bytecode instructions"""
    bytecode = dis.Bytecode(method).dis()
    return len([line for line in bytecode.splitlines() if line])


def get_reward_from_bitstring(s: str) -> float:
    """Calculate reward from bit string"""
    reward = 0
    sign = 1 if s[0] == "0" else -1
    for idx in range(len(s) - 1):
        reward += int(s[idx + 1]) * pow(0.5, idx - 1)
    return sign * reward


def get_random_bit() -> int:
    """Returns a pseudorandom bit from a timestamp"""
    seed = time.time()
    return pow(2, int(str(seed).replace('.', '')[-5:])) % 3 % 2


def get_resources_path() -> Path:
    """Returns resources path."""
    return Path(__file__).parent.joinpath('resources')


def agents() -> List[type(Agent)]:
    """Initialize and return a list of agents"""
    return [
        RandomActions,
        Handcrafted,
        NNrelu0,
        NNrelu4443
    ]


def environments() -> List[type(Environment)]:
    """Initialize and return a list of environments"""
    return [
        BiasedCoinFlip,
        DoubleCoinFlip,
        Slide,
        AnyOne,
        SpecificOne
    ]
