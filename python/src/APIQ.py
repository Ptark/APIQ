import concurrent.futures
import importlib
import pickle
import pprint
from pathlib import Path
from typing import Type, Tuple
from python.src import Utility

from python.src.agents.abstract_classes.Agent import Agent
from python.src.environments.abstract_classes.Environment import Environment

number_of_cycles = 100000

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
        path = data_dir_path.joinpath(ag + "_" + env + ".apiq")
        if path.is_file():
            rewards = pickle.load(path.open("rb"))
            Utility.nested_set(reward_dict, [ag, env], rewards)


def apiq():
    """Trials agents in environments and calculates apiq from results"""
    print("----------------------------------------")
    print("Trial agents in environments")
    trials()
    pprint.pprint(reward_dict)
    for ag_name in reward_dict:
        for env_name in reward_dict[ag_name]:
            data_path = data_dir_path.joinpath(ag_name + "_" + env_name + ".apiq")
            pickle.dump(reward_dict[ag_name][env_name], data_path.open("wb"))
    print("----------------------------------------")
    print("Calculate scaling factors")
    environment_scaling_factors = calculate_scaling_factors()
    pprint.pprint(environment_scaling_factors)
    print("----------------------------------------")
    print("Calculate apiq")
    apiq_dict = {}
    norming_factor = sum(environment_scaling_factors.values())
    for ag_name in reward_dict:
        ag_apiq = 0
        for env_name in reward_dict[ag_name]:
            positive = reward_dict[ag_name][env_name]["0"]
            negative = reward_dict[ag_name][env_name]["1"]
            ag_apiq += (positive + negative) * environment_scaling_factors[env_name]
        ag_apiq /= norming_factor
        apiq_dict[ag_name] = ag_apiq
    pprint.pprint(apiq_dict)
    print("----------------------------------------")
    pickle.dump(apiq_dict, apiq_dict_path.open("wb"))


def trials():
    """Trials agents in environments"""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_list = []
        for pair in pairs_to_be_trialed():
            future_list.append(executor.submit(trial_agent_environment, pair, "0"))
            future_list.append(executor.submit(trial_agent_environment, pair, "1"))
        for future in concurrent.futures.as_completed(future_list):
            ag_name, env_name, reward, sign_bit = future.result()
            Utility.nested_set(reward_dict, [ag_name, env_name, sign_bit], reward)


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
    total_reward /= number_of_cycles
    print("    %s in %s done." % (agent_class.__name__, environment_class.__name__))
    return agent_class.__name__, environment_class.__name__, total_reward, sign_bit


def pairs_to_be_trialed() -> set:
    """Returns a set of agent environment pairs which have to be trialed"""
    agent_environment_pairs = set()
    for agent_class in agent_classes:
        ag_name = agent_class.__name__
        reward_dict.setdefault(ag_name, {})
        for environment_class in environment_classes:
            if environment_class.__name__ not in reward_dict[ag_name]:
                agent_environment_pairs.add((agent_class, environment_class))
    return agent_environment_pairs


def calculate_scaling_factors() -> dict:
    """Calculates scaling factors for all environments and saves them in a dictionary"""
    environment_scaling_factors = {}
    for environment_class in environment_classes:
        scaling_factor = Utility.get_scaling_factor(environment_class())
        environment_scaling_factors[environment_class.__name__] = scaling_factor
    return environment_scaling_factors
