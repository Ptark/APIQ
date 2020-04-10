from typing import List, Tuple

from python.src import Utility
from python.src.agents.Agent import Agent
from python.src.environments.Environment import Environment


def apiq_scores(number_of_evaluations: int) -> List:
    """calculate APIQ scores for all agents and accumulate results in a dictionary"""
    apiq = []
    agents = Utility.agents()
    environments = Utility.environments()
    scaling_factors = [Utility.get_scaling_factor(environment()) for environment in environments]
    for agent in agents:
        counter = 0
        denominator = 0
        agent_dict = {
            "name": agent.__name__,
            "environment_scores": []
        }
        for idx in range(len(environments)):
            reward_positive = evaluate_agent_environment((agent(), environments[idx]()), number_of_evaluations, "0")
            reward_negative = evaluate_agent_environment((agent(), environments[idx]()), number_of_evaluations, "1")
            counter += (reward_positive + reward_negative) * scaling_factors[idx]
            denominator += scaling_factors[idx]
            environment_dict = {
                "name": environments[idx].__name__,
                "reward_positive": reward_positive,
                "reward_negative": reward_negative
            }
            agent_dict["environment_scores"].append(environment_dict)
        agent_dict["apiq"] = counter / denominator
        apiq.append(agent_dict)
    return apiq


def evaluate_agent_environment(agent_environment_pair: Tuple[Agent, Environment], number_of_evaluations: int,
                               sign_bit: str) -> float:
    """Evaluate the reward an agent earns on average in an environment"""
    total_reward = 0
    agent = agent_environment_pair[0]
    environment = agent_environment_pair[1]
    for i in range(number_of_evaluations):
        reward = 0
        percept = (str(agent.idx), sign_bit)
        for turns in range(environment.turns):
            action = agent.calculate_action(percept)
            percept = environment.calculate_percept(action)
            for idx in range(len(percept[1])):
                reward += int(percept[1][idx]) * pow(0.5, idx)
        sign = 1 if sign_bit == "0" else -1
        total_reward += sign * reward
    return total_reward / number_of_evaluations
