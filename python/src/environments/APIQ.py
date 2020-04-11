from typing import List, Type
from python.src import Utility
from python.src.agents.Agent import Agent
from python.src.environments.Environment import Environment


def complexity() -> List:
    """Return a dictionary which holds all environments and their complexity"""
    complexity_dict = []
    for environment in Utility.environments():
        env = environment()
        complexity_dict.append({
            "name": environment.__name__,
            "complexity": Utility.environment_complexity(env, env.randomness)
        })
    return complexity_dict


def apiq(number_of_evaluations: int) -> List:
    """calculate APIQ scores for all agents and accumulate results in a dictionary"""
    apiq_dict = []
    agents = Utility.agents()
    for agent in agents:
        apiq_dict.append(apiq_agent(agent, number_of_evaluations))
    return apiq_dict


def apiq_agent(agent, number_of_evaluations):
    """calculate APIQ for an agent and accumulate results in a dictionary"""
    environments = Utility.environments()
    scaling_factors = [Utility.get_scaling_factor(env(), env().randomness) for env in environments]
    counter = 0
    denominator = 0
    agent_dict = {
        "name": agent.__name__,
        "environment_scores": []
    }
    for idx in range(len(environments)):
        reward_positive = reward_agent_environment(agent, environments[idx], number_of_evaluations, "0")
        reward_negative = reward_agent_environment(agent, environments[idx], number_of_evaluations, "1")
        counter += (reward_positive + reward_negative) * scaling_factors[idx]
        denominator += scaling_factors[idx]
        environment_dict = {
            "name": environments[idx].__name__,
            "reward_positive": reward_positive,
            "reward_negative": reward_negative
        }
        agent_dict["environment_scores"].append(environment_dict)
    agent_dict["apiq"] = counter / denominator
    return agent_dict


def reward_agent_environment(agent_class: Type[Agent], environment_class: Type[Environment], number_of_evaluations: int,
                             sign_bit: str) -> float:
    """Evaluate the reward an agent earns on average in an environment"""
    total_reward = 0
    for i in range(number_of_evaluations):
        agent = agent_class()
        environment = environment_class()
        reward = 0
        percept = (str(environment.idx), sign_bit)
        for turns in range(environment.number_of_turns):
            action = agent.calculate_action(percept)
            percept = environment.calculate_percept(action)
            for idx in range(len(percept[1])):
                reward += int(percept[1][idx]) * pow(0.5, idx)
        sign = 1 if sign_bit == "0" else -1
        total_reward += sign * reward
    return total_reward / number_of_evaluations
