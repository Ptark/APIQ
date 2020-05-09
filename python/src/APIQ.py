import concurrent.futures
import pickle
import pprint
from pathlib import Path
from typing import Type

from python.src import Utility

from python.src.agents.abstract_classes.Agent import Agent
from python.src.agents.Handcrafted import Handcrafted
from python.src.agents.NNtanh import NNtanh
from python.src.agents.NNrelu import NNrelu
from python.src.agents.NNrelu4 import NNrelu4
from python.src.agents.NNsigmoid import NNsigmoid
from python.src.agents.RandomActions import RandomActions
from python.src.agents.abstract_classes.TrainableAgent import TrainableAgent

from python.src.environments.abstract_classes.Environment import Environment
from python.src.environments.Slide import Slide
from python.src.environments.SlideR import SlideR
from python.src.environments.AnyOne import AnyOne
from python.src.environments.AnyOneR import AnyOneR
from python.src.environments.SpecificOne import SpecificOne
from python.src.environments.SpecificOneR import SpecificOneR
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip
from python.src.environments.BiasedCoinFlipR import BiasedCoinFlipR
from python.src.environments.DoubleCoinFlip import DoubleCoinFlip
from python.src.environments.DoubleCoinFlipR import DoubleCoinFlipR

number_of_evaluations = 1000
training_steps = 1001

data_dir_path = Utility.get_data_path()
Path(data_dir_path).mkdir(parents=True, exist_ok=True)
environment_scaling_factors = {}
apiq_dict_path = data_dir_path.joinpath('apiq_dict.apiq')
apiq_dict = pickle.load(apiq_dict_path.open("rb")) if apiq_dict_path.is_file() else {}
agent_classes = [
    RandomActions,
    Handcrafted,
    NNrelu,
    NNrelu4,
    NNsigmoid,
    NNtanh
]
environment_classes = [
    BiasedCoinFlip,
    BiasedCoinFlipR,
    DoubleCoinFlip,
    DoubleCoinFlipR,
    Slide,
    SlideR,
    AnyOne,
    AnyOneR,
    SpecificOne,
    SpecificOneR,
]


def calculate_scaling_factors():
    """Calculates scaling factors for all environments and saves them in a dictionary"""
    print("Calculating scaling factors...")
    for environment_class in environment_classes:
        scaling_factor = Utility.get_scaling_factor(environment_class())
        environment_scaling_factors[environment_class.__name__] = scaling_factor


def train():
    """Train trainable agents up to training_steps and save results"""
    print("Training...")
    agent_environment_pairs = set()
    for agent_class in agent_classes:
        apiq_dict.setdefault(agent_class.__name__, {})
        for environment_class in environment_classes:
            ag_name, env_name = agent_class.__name__, environment_class.__name__
            apiq_dict[ag_name].setdefault(env_name, {
                "training_steps": {
                    0: None
                }
            })
            if issubclass(agent_class, TrainableAgent):
                agent_environment_pairs.add((agent_class, environment_class))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_list = []
        for pair in agent_environment_pairs:
            max_step = max(apiq_dict[pair[0].__name__][pair[1].__name__]["training_steps"])
            future_list.append(executor.submit(train_agent_environment, pair[0], pair[1], max_step))
        for future in concurrent.futures.as_completed(future_list):
            training_step_list, ag_key, env_key = future.result()
            for training_step in training_step_list:
                key_list = [ag_key, env_key, "training_steps", training_step]
                Utility.nested_set(apiq_dict, key_list, None)


def train_agent_environment(agent_class: Type[TrainableAgent], environment_class: Type[Environment], max_step: int):
    """Train trainable agent in environment until training_steps is reached"""
    training_step_list = []
    ag_name, env_name = agent_class.__name__, environment_class.__name__
    path = data_dir_path.joinpath(ag_name + "/" + env_name + "/" + str(max_step) + ".apiq") if max_step != 0 else ''
    agent = agent_class(environment_class=environment_class, path=path)
    for training_step in range(max_step, training_steps):
        environment = environment_class()
        observation = "1" * environment.observation_length
        reward = environment.sign_bit + "0" * (environment.reward_length - 1)
        percept = (observation, reward)
        for turns in range(environment.number_of_turns):
            action = agent.calculate_action(percept)
            percept = environment.calculate_percept(action)
            agent.train(percept[1])
        if Utility.is_saved(training_step):
            dir_path = data_dir_path.joinpath(ag_name + "/" + env_name)
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            path = dir_path.joinpath(str(training_step) + ".apiq")
            agent.save(path)
            training_step_list.append(training_step)
        agent.reset()
    return training_step_list, ag_name, env_name


def calculate_rewards():
    """Calculates the rewards the agents receive in the environments"""
    print("Calculating reward...")
    agent_environment_step_tuples = set()
    for agent_class in agent_classes:
        for environment_class in environment_classes:
            ag_name, env_name = agent_class.__name__, environment_class.__name__
            for training_step in apiq_dict[ag_name][env_name]["training_steps"]:
                if apiq_dict[ag_name][env_name]["training_steps"][training_step] is None:
                    agent_environment_step_tuples.add((agent_class, environment_class, training_step))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_list = {executor.submit(rewards_agent_environment_step, triple[0], triple[1], triple[2]): triple for triple in agent_environment_step_tuples}
        for future in concurrent.futures.as_completed(future_list):
            average_reward, ag_key, env_key, training_step = future.result()
            Utility.nested_set(apiq_dict, [ag_key, env_key, "training_steps", training_step], average_reward)


def rewards_agent_environment_step(agent_class: Type[Agent], environment_class: Type[Environment], training_step: int):
    """Calculates average reward for an agent in an environment at training_step
    Parallel execution possible"""
    ag_name, env_name = agent_class.__name__, environment_class.__name__
    summed_reward = 0
    evaluations = number_of_evaluations
    path = data_dir_path.joinpath(ag_name + "/" + env_name + "/" + str(training_step) + ".apiq") if training_step != 0 else ''
    if not environment_class.has_randomness and not agent_class.has_randomness:
        evaluations = 1
    for idx in range(evaluations):
        agent = agent_class(environment_class, path) if issubclass(agent_class, TrainableAgent) else agent_class(environment_class)
        environment = environment_class()
        observation = "1" * environment_class.observation_length
        reward = environment_class.sign_bit + "0" * (environment_class.reward_length - 1)
        percept = (observation, reward)
        for turns in range(environment_class.number_of_turns):
            action = agent.calculate_action(percept)
            percept = environment.calculate_percept(action)
            summed_reward += Utility.get_reward_from_bitstring(percept[1])
    average_reward = summed_reward / evaluations
    return average_reward, ag_name, env_name, training_step


def calculate_apiq():
    """Calculates APIQ for all agents from apiq_dict"""
    print("Calculating APIQ...")
    pprint.pprint(apiq_dict)
    norming_factor = sum(environment_scaling_factors.values())

    for agent_class in agent_classes:
        apiq_dict[agent_class.__name__]["apiq"] = {}
        ag_name = agent_class.__name__
        for environment_class in environment_classes:
            env_name = environment_class.__name__
            for training_step in apiq_dict[ag_name][env_name]["training_steps"]:
                apiq_dict[ag_name]["apiq"].setdefault(training_step, 0)
                reward = apiq_dict[ag_name][env_name]["training_steps"][training_step]
                apiq_dict[ag_name]["apiq"][training_step] += reward * environment_scaling_factors[env_name]
    for agent_class in agent_classes:
        ag_name = agent_class.__name__
        for training_step in apiq_dict[ag_name]["apiq"]:
            apiq_dict[ag_name]["apiq"][training_step] /= norming_factor
    path = data_dir_path.joinpath("apiq_dict.apiq")
    pickle.dump(apiq_dict, path.open("wb"))

