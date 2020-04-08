from typing import List

from python.src.agents.Agent import Agent
from python.src.agents.RandomAgent import RandomAgent
from python.src.environments.Environment import Environment
from python.src.environments.Slide import Slide
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip
from python.src.environments.DoubleCoinFlip import DoubleCoinFlip


def agents() -> List[Agent]:
    """Initialize and return a list of agents"""
    return [RandomAgent()]


def environments() -> List[Environment]:
    """Initialize and return a list of environments"""
    return [
        BiasedCoinFlip(),
        DoubleCoinFlip(),
        Slide(),
    ]


def evaluate_agent_in_environment(ag: Agent, env: Environment, num_eval: int, sign: int) -> float:
    """Evaluate a given agent in the given environment a given number of times
    Return: Arithmetic mean of earned reward
    """
    total_reward = 0
    for i in range(num_eval):
        reward = 0
        percept = (env.idx, sign)
        for turns in range(env.turns):
            action = ag.calculate_action(percept)
            percept = env.calculate_percept(action)
            count = 0
            for x in percept[1]:
                reward += x * pow(0.5, count)
                count += 1
        total_reward += sign * reward
    return total_reward / number_of_evaluations


number_of_evaluations = 10000         # number of evaluations of an agent in an environment for stochastic purposes
agents = agents()                     # list of agents
scores = []
for agent in agents:                  # loop for evaluating all agents
    environments = environments()     # initialize list of environments
    for environment in environments:  # loop for evaluating agent in all environments
        scores.append(evaluate_agent_in_environment(agent, environment, number_of_evaluations, 1))
        scores.append(evaluate_agent_in_environment(agent, environment, number_of_evaluations, -1))
print(scores)




