import concurrent.futures
import importlib
import math
import pickle
import pprint
from pathlib import Path
from typing import Type, Tuple

import numpy as np

from python.src import Utility
from statistics import NormalDist

from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment

number_of_cycles = 10000
number_of_trials = 100
floating_precision_factor = 25

agent_module = importlib.import_module("python.src.agents")
environment_module = importlib.import_module("python.src.environments")
agent_classes = {getattr(agent_module, class_name) for class_name in agent_module.__all__}
environment_classes = {getattr(environment_module, class_name) for class_name in environment_module.__all__}

data_dir_path = Utility.get_data_path()
Path(data_dir_path).mkdir(parents=True, exist_ok=True)
apiq_dict_path = data_dir_path.joinpath('apiq_dict.apiq')

# Loads already evaluated agent-environment combinations into reward dict
reward_dict = {}
for ag_class in agent_classes:
    ag = ag_class.__name__
    for env_class in environment_classes:
        env = env_class.__name__
        Utility.nested_set(reward_dict, [ag, env, "positive", "rewards"], [])
        Utility.nested_set(reward_dict, [ag, env, "negative", "rewards"], [])
        path = data_dir_path.joinpath(ag + "_" + env + ".apiq")
        if path.is_file():
            rewards = pickle.load(path.open("rb"))
            Utility.nested_set(reward_dict, [ag, env], rewards)


def apiq():
    """Trials agents in environments and calculates apiq from results"""
    complexity_dict = calculate_complexities()
    complexity_dict = {k: v for k, v in sorted(complexity_dict.items(), key=lambda item: item[1])}
    scaling_factor_dict, discrete_distribution, continuous_distribution = calculate_scaling_factors(complexity_dict)
    print("----------------------------------------")
    print("Trialing agents in environments...")
    trials()
    for ag_name in reward_dict:
        for env_name in reward_dict[ag_name]:
            for sign in ["positive", "negative"]:
                collected_rewards = reward_dict[ag_name][env_name][sign]["rewards"]
                reward_dict[ag_name][env_name][sign]["mean"] = np.mean(collected_rewards)
                reward_dict[ag_name][env_name][sign]["error"] = np.std(collected_rewards, dtype=np.float64, ddof=1)
    apiq_dict = {}
    norming_factor = sum(scaling_factor_dict.values())
    for ag_name in reward_dict:
        apiq_mean = 0
        apiq_error = 0
        for env_name in reward_dict[ag_name]:
            ag_env_dict = reward_dict[ag_name][env_name]
            factor = scaling_factor_dict[env_name]
            positive_mean = ag_env_dict["positive"]["mean"]
            p_err = ag_env_dict["positive"]["error"]
            negative_mean = ag_env_dict["negative"]["mean"]
            n_err = ag_env_dict["negative"]["error"]
            apiq_mean += (positive_mean + negative_mean) * factor
            apiq_error += (p_err * p_err + n_err * n_err) * factor * factor
        apiq_mean /= norming_factor
        apiq_error = np.sqrt(apiq_error) / norming_factor
        Utility.nested_set(apiq_dict, [ag_name, "mean"], apiq_mean)
        Utility.nested_set(apiq_dict, [ag_name, "error"], apiq_error)
    # sort dictionaries, print them and save them
    #   sort complexity_dict by complexity
    print("----------------------------------------")
    print("Complexitiy - Number of Bytecode instructions:")
    for k in complexity_dict:
        print("    {:s}: {:d}".format(k, complexity_dict[k]))
    complexity_dict_path = data_dir_path.joinpath("complexity_dict.apiq")
    pickle.dump(complexity_dict, complexity_dict_path.open("wb"))
    print("----------------------------------------")
    print("Scaling Factor - Inverse of density function:")
    for k in scaling_factor_dict:
        print("    {:s}: {:f}".format(k, scaling_factor_dict[k]))
    scaling_factor_dict_path = data_dir_path.joinpath("scaling_factor_dict.apiq")
    pickle.dump(complexity_dict, scaling_factor_dict_path.open("wb"))
    discrete_distribution_path = data_dir_path.joinpath("discrete_distribution.apiq")
    pickle.dump(discrete_distribution, discrete_distribution_path.open("wb"))
    continuous_distribution_path = data_dir_path.joinpath("continuous_distribution.apiq")
    pickle.dump(continuous_distribution, continuous_distribution_path.open("wb"))

    #   print reward_dict - PiAgents first, then alphabetically
    print("----------------------------------------")
    print("Positive and negative rewards:")
    for ag_name in reward_dict:
        for env_name in reward_dict[ag_name]:
            data_path = data_dir_path.joinpath(ag_name + "_" + env_name + ".apiq")
            pickle.dump(reward_dict[ag_name][env_name], data_path.open("wb"))
    agent_list = ["PiRand", "PiBasic", "Pi2Back", "Pi2Forward", "Handcrafted", "NNsigmoid", "NNsigmoid4", "NNrelu",
                  "NNrelu4", "NNreluSigmoid", "NNrelu4Sigmoid"]
    agent_list.extend([k for k in reward_dict if k not in agent_list])
    Utility.sort_dict(reward_dict, agent_list)
    environment_list = [k for k in complexity_dict]
    for a in reward_dict:
        Utility.sort_dict(reward_dict[a], environment_list)
    for a in reward_dict:
        print("    {}:".format(a))
        for e in reward_dict[a]:
            temp = reward_dict[a][e]
            positive = temp["positive"]["mean"]
            negative = temp["negative"]["mean"]
            p_err = temp["positive"]["error"]
            n_err = temp["negative"]["error"]
            print("        {:s}: {:.5f} +- {:.5f}, {:.5f} +- {:.5f}".format(e, positive, p_err, negative, n_err))
    reward_dict_path = data_dir_path.joinpath("reward_dict.apiq")
    pickle.dump(reward_dict, reward_dict_path.open("wb"))
    #    print apiq_dict - PiAgents first, then alphabetically
    print("----------------------------------------")
    print("APIQ:")
    Utility.sort_dict(apiq_dict, agent_list)
    for a in apiq_dict:
        print("    {:s}: {:.5f} +- {:.5f}".format(a, apiq_dict[a]["mean"], apiq_dict[a]["error"]))
    pickle.dump(apiq_dict, apiq_dict_path.open("wb"))


def trials():
    """Trials agents in environments"""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_list = []
        for pair in pairs_to_be_trialed():
            for idx in range(number_of_trials):
                future_list.append(executor.submit(trial_agent_environment, pair, "0"))
                future_list.append(executor.submit(trial_agent_environment, pair, "1"))
        idx = 1
        for future in concurrent.futures.as_completed(future_list):
            ag_name, env_name, reward, sign = future.result()
            reward_dict[ag_name][env_name][sign]["rewards"].append(reward)
            print("    {:d}/{:d} done.".format(idx, len(future_list)))
            idx += 1


def trial_agent_environment(pair: Tuple[Type[Agent], Type[Environment]], sign_bit: str) -> Tuple[str, str, float, str]:
    """Trial agent in environment"""
    agent_class, environment_class = pair[0], pair[1]
    total_reward = 0
    environment = environment_class(sign_bit)
    agent = agent_class(environment=environment)
    observation = "0" * environment.observation_length
    for i in range(number_of_cycles):
        action = agent.calculate_action(observation)
        observation, reward = environment.calculate_percept(action)
        total_reward += Utility.get_reward_from_bitstring(reward)
        agent.train(reward)
    total_reward /= number_of_cycles * environment_class.max_average_reward_per_cycle
    sign = "positive" if sign_bit == "0" else "negative"
    return agent_class.__name__, environment_class.__name__, total_reward, sign


def pairs_to_be_trialed() -> set:
    """Returns a set of agent environment pairs which have to be trialed"""
    agent_environment_pairs = set()
    for agent_class in agent_classes:
        ag_name = agent_class.__name__
        reward_dict.setdefault(ag_name, {})
        for environment_class in environment_classes:
            if len(reward_dict[ag_name][environment_class.__name__]["positive"]["rewards"]) == 0:
                agent_environment_pairs.add((agent_class, environment_class))
    return agent_environment_pairs


def calculate_complexities() -> dict:
    """Calculates complexities for all environments and saves them in a dictionary"""
    environment_complexities = {}
    for environment_class in environment_classes:
        complexity = Utility.calculate_complexity(environment_class())
        environment_complexities[environment_class.__name__] = complexity
    return environment_complexities


def calculate_scaling_factors(complexity_dict: dict) -> Tuple[dict, list, list]:
    """Calculates scaling factors for all environments from complexities and saves them in a dictionary.
    Parameters:
        complexity_dict: The dictionary with the environments as keys and their complexities as values
    Returns:
        scaling_factor_dict: Dictionary with environments as keys and their scaling factors as values.
        discrete_distribution: Discrete distribution of complexity of environments.
        continuous distribution: Continuous distribution of complexity of environments."""
    complexities = complexity_dict.values()
    max_c = max(complexities)
    min_c = min(complexities)
    discrete_distribution = [0] * (max_c + 1)
    for value in complexity_dict.values():
        discrete_distribution[value] += 1
    sigma = math.sqrt((max_c - min_c + 1) / len(complexities))
    normal_distribution = [NormalDist(0, sigma).pdf(x) for x in range(math.floor(-3 * sigma), math.ceil(3 * sigma))]
    half_len = math.floor(len(normal_distribution) / 2)
    continuous_distribution = np.convolve(discrete_distribution, normal_distribution)
    continuous_distribution = continuous_distribution[half_len:-half_len + 1]
    scaling_factor_dict = {}
    for k, v in complexity_dict.items():
        scaling_factor_dict[k] = 1 / continuous_distribution[v]
    return scaling_factor_dict, discrete_distribution, continuous_distribution
