from typing import List

from python.src import Utility
from python.src.agents.Agent import Agent
from python.src.environments.Environment import Environment


def calculate_apiq_scores(number_of_evaluations: int) -> List:
    """calculate APIQ scores for all agents and accumulate results in a dictionary"""
    apiq_scores = []
    agents = Utility.agents()  # list of agents
    # calculate complexity scaling factors
    scaling_factors = [Utility.get_scaling_factor(env) for env in Utility.environments()]
    for agent in agents:  # loop for evaluating all agents
        apiq_scores.append(evaluate_agent(agent, number_of_evaluations, scaling_factors))
    return apiq_scores


def evaluate_agent(agent: Agent, number_of_evaluations: int, scaling_factors: List[float]) -> dict:
    """evaluate an agents APIQ"""
    environments = Utility.environments()     # initialize list of environments
    counter = 0
    denominator = 0
    idx = 0
    agent_dict = {
        "name": type(agent).__name__,
        "environment_scores": []
    }
    for environment in environments:  # loop for evaluating agent in all environments
        positive_score = evaluate_agent_in_environment(agent, environment, number_of_evaluations, 1)
        negative_score = evaluate_agent_in_environment(agent, environment, number_of_evaluations, -1)
        counter += (positive_score + negative_score) * scaling_factors[idx]
        denominator += 2 * scaling_factors[idx]
        idx += 1
        environment_dict = {
            "name": type(environment).__name__,
            "positive_score": positive_score,
            "negative_score": negative_score
        }
        agent_dict["environment_scores"].append(environment_dict)
    agent_dict["apiq"] = counter / denominator
    return agent_dict


def evaluate_agent_in_environment(agent: Agent, environment: Environment, number_of_evaluations: int, sign: int) -> float:
    """Evaluate a given agent in the given environment a given number of times
    Return: Arithmetic mean of earned reward
    """
    total_reward = 0
    for i in range(number_of_evaluations):
        reward = 0
        percept = (environment.idx, sign)
        for turns in range(environment.turns):
            action = agent.calculate_action(percept)
            percept = environment.calculate_percept(action)
            for idx in range(len(percept[1])):
                reward += int(percept[1][idx]) * pow(0.5, idx)
        total_reward += sign * reward
    return total_reward / number_of_evaluations
