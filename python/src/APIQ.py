from typing import List, Type

import numpy as np

from python.src import Utility
from python.src.agents.Agent import Agent
from python.src.environments.Environment import Environment


def apiq(number_of_evaluations: int) -> List:
    """calculate APIQ scores for all agents and accumulate results in a dictionary"""
    apiq_dict = []
    agents = Utility.agents()
    for agent in agents:
        apiq_dict.append(apiq_agent(agent, number_of_evaluations))
    return apiq_dict


def apiq_agent(agent_class: Type[Agent], number_of_evaluations):
    """calculate APIQ for an agent and accumulate results in a dictionary"""
    training_steps = agent_class.training_steps
    environments = Utility.environments()
    scaling_factors = [Utility.get_scaling_factor(env(), env().randomness) for env in environments]
    norming_factor = sum(scaling_factors)
    sum_scaled_rewards = []
    for step in range(training_steps):
        if step == 0 or np.log10(step).is_integer():
            sum_scaled_rewards.append(0)
    agent_dict = {
        "name": agent_class.__name__,
        "environment_scores": []
    }
    for idx in range(len(environments)):
        environment_dict = {
            "name": environments[idx].__name__,
            "reward-positive": [],
            "reward-negative": [],
            "step": []
        }
        for step in range(training_steps):
            if step == 0 or np.log10(step).is_integer():
                reward_positive = reward_agent_environment(agent_class, environments[idx], number_of_evaluations, "0",
                                                           step)
                reward_negative = reward_agent_environment(agent_class, environments[idx], number_of_evaluations, "1",
                                                           step)
                environment_dict["reward-positive"].append(reward_positive)
                environment_dict["reward-negative"].append(reward_negative)
                environment_dict["step"].append(step)
                sum_scaled_rewards[step] += (reward_positive + reward_negative) * scaling_factors[idx]
        agent_dict["environment_scores"].append(environment_dict)
    agent_dict["apiq"] = []
    for unnormalized_apiq in sum_scaled_rewards:
        agent_dict["apiq"].append(unnormalized_apiq / norming_factor)
    return agent_dict


def reward_agent_environment(agent_class: Type[Agent], environment_class: Type[Environment], number_of_evaluations: int,
                             sign_bit: str, step: int) -> float:
    """Evaluate the reward an agent earns on average in an environment"""
    summed_reward = 0
    for i in range(number_of_evaluations):
        environment = environment_class()
        agent = agent_class(step)
        reward = 0
        percept = (str(environment.idx), sign_bit)
        for turns in range(environment.number_of_turns):
            action = agent.calculate_action(percept)
            percept = environment.calculate_percept(action)
            reward += Utility.get_reward_from_bitstring(percept[1])
        sign = 1 - 2 * (int(sign_bit) == 1)
        summed_reward += sign * reward
    return summed_reward / number_of_evaluations


def train_agent_environment(agent_class: Type[Agent], environment_class: Type[Environment]):
    """Train agent in environment for sign = 1 and sign = -1"""
    agent_class(0).train(environment_class, "0")
    agent_class(0).train(environment_class, "1")


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
