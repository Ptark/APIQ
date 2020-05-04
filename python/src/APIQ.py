import pickle
from typing import List, Type

import numpy as np

from python.src import Utility
from python.src.agents.Agent import Agent
from python.src.environments.Environment import Environment

# load set of trained agent environment tuples
is_trained_set_path = Utility.get_resources_path().joinpath('training_data')
is_trained_set = pickle.load(is_trained_set_path.open("rb"))


def apiq(number_of_evaluations: int, training_steps: int, step_size: int) -> List:
    """calculate APIQ scores for all agents and accumulate results in a dictionary"""
    apiq_dict = []
    agents = Utility.agents()
    for agent in agents:
        apiq_dict.append(apiq_agent(agent, number_of_evaluations, training_steps, step_size))
    return apiq_dict


def apiq_agent(agent_class: Type[Agent], number_of_evaluations: int, training_steps: int, step_size: int):
    """calculate APIQ for an agent and accumulate results in a dictionary"""
    environments = Utility.environments()
    scaling_factors = [Utility.get_scaling_factor(env("0"), env("0").randomness) for env in environments]
    norming_factor = sum(scaling_factors)
    sum_scaled_rewards = []
    for step in range(training_steps):
        if step % step_size == 0:
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
            if step == 0 or (step % step_size == 0 and agent_class.is_trainable):
                reward_positive = reward_agent_environment(agent_class, environments[idx], number_of_evaluations, "0",
                                                           step)
                reward_negative = reward_agent_environment(agent_class, environments[idx], number_of_evaluations, "1",
                                                           step)
                environment_dict["reward-positive"].append(reward_positive)
                environment_dict["reward-negative"].append(reward_negative)
                environment_dict["step"].append(step)
                sum_scaled_rewards[int(step / step_size)] += (reward_positive + reward_negative) * scaling_factors[idx]
        agent_dict["environment_scores"].append(environment_dict)
    agent_dict["apiq"] = []
    for unnormalized_apiq in sum_scaled_rewards:
        agent_dict["apiq"].append(unnormalized_apiq / norming_factor)
    return agent_dict


def reward_agent_environment(agent_class: Type[Agent], environment_class: Type[Environment], number_of_evaluations: int,
                             sign: str, training_step: int) -> float:
    """Evaluate the reward an agent earns on average in an environment"""
    summed_reward = 0
    for i in range(number_of_evaluations):
        environment = environment_class(sign)
        agent = agent_class(environment, training_step)
        observation = "1" * environment.observation_length
        reward = "0" * environment.reward_length
        percept = (observation, reward)
        for turns in range(environment.number_of_turns):
            action = agent.calculate_action(percept)
            percept = environment.calculate_percept(action)
            summed_reward += Utility.get_reward_from_bitstring(percept[1])
    return summed_reward / number_of_evaluations


def train():
    """Train trainable untrained agents"""
    agent_classes = Utility.agents()
    for agent_class in agent_classes:
        if agent_class.is_trainable:
            train_agent(agent_class)
    # save set of trained class environment tuples
    pickle.dump(is_trained_set, is_trained_set_path.open("wb"))


def train_agent(agent_class: Type[Agent]):
    """Train agent in environments"""
    for environment_class in Utility.environments():
        if not (agent_class, environment_class) in is_trained_set:
            train_agent_environment(agent_class, environment_class)
            is_trained_set.add((agent_class, environment_class))


def train_agent_environment(agent_class: Type[Agent], environment_class: Type[Environment]):
    """Train agent in environment for sign = 1 and sign = -1"""
    agent_class(environment_class(1), 0).train()
    agent_class(environment_class(-1), 0).train()


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
