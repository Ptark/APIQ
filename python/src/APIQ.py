import pickle
from typing import List, Type

from python.src import Utility

from python.src.agents.Agent import Agent
from python.src.agents.Handcrafted import Handcrafted
from python.src.agents.NNtanh import NNtanh
from python.src.agents.NNrelu import NNrelu
from python.src.agents.NNrelu4 import NNrelu4
from python.src.agents.NNsigmoid import NNsigmoid
from python.src.agents.RandomActions import RandomActions

from python.src.environments.Environment import Environment
from python.src.environments.Slide import Slide
from python.src.environments.AnyOne import AnyOne
from python.src.environments.SpecificOne import SpecificOne
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip
from python.src.environments.DoubleCoinFlip import DoubleCoinFlip

is_trained_set_path = Utility.get_resources_path().joinpath('data/is_trained_set.apiq')
is_evaluated_set_path = Utility.get_resources_path().joinpath('data/is_evaluated_set.apiq')
is_trained_set = pickle.load(is_trained_set_path.open("rb")) if is_trained_set_path.is_file() else set()
number_of_evaluations = 1000
training_steps = 1001
step_size = 200


def apiq() -> List:
    """calculate APIQ scores for all agents and accumulate results in a dictionary"""
    print("Evaluate APIQ...")
    apiq_dict = []
    agent_classes = get_agents()
    for idx in range(len(agent_classes)):
        print("Evaluate %s %d/%d" % (agent_classes[idx].__name__, idx + 1, len(agent_classes)))
        apiq_dict.append(apiq_agent(agent_classes[idx]))
    return apiq_dict


def apiq_agent(agent_class: Type[Agent]):
    """calculate APIQ for an agent and accumulate results in a dictionary"""
    environment_classes = get_environments()
    scaling_factors = [Utility.get_scaling_factor(env("0"), env("0").randomness) for env in environment_classes]
    norming_factor = sum(scaling_factors)
    sum_scaled_rewards = []
    for training_step in range(training_steps):
        if training_step % step_size == 0:
            sum_scaled_rewards.append(0)
    agent_dict = {
        "name": agent_class.__name__,
        "environment_scores": []
    }
    for idx in range(len(environment_classes)):
        print("    Training on %s - (%d/%d)" % (environment_classes[idx].__name__, idx + 1, len(environment_classes)))
        environment_dict = {
            "name": environment_classes[idx].__name__,
            "reward-positive": [],
            "reward-negative": [],
            "step": []
        }
        for training_step in range(training_steps):
            if training_step == 0 or (training_step % step_size == 0 and agent_class.is_trainable):
                reward_positive = reward_agent_environment(agent_class, environment_classes[idx], "0",
                                                           training_step)
                reward_negative = reward_agent_environment(agent_class, environment_classes[idx], "1",
                                                           training_step)
                environment_dict["reward-positive"].append(reward_positive)
                environment_dict["reward-negative"].append(reward_negative)
                environment_dict["step"].append(training_step)
                sum_scaled_rewards[int(training_step / step_size)] += (reward_positive + reward_negative) * scaling_factors[idx]
        agent_dict["environment_scores"].append(environment_dict)
    agent_dict["apiq"] = []
    for unnormalized_apiq in sum_scaled_rewards:
        agent_dict["apiq"].append(unnormalized_apiq / norming_factor)
    return agent_dict


def reward_agent_environment(agent_class: Type[Agent], environment_class: Type[Environment], sign_bit: str,
                             training_step: int) -> float:
    """Evaluate the reward an agent earns on average in an environment"""
    summed_reward = 0
    for i in range(number_of_evaluations):
        agent = agent_class(environment_class, sign_bit, training_step)
        environment = environment_class(sign_bit)
        observation = "1" * environment.observation_length
        reward = sign_bit + "0" * (environment.reward_length - 1)
        percept = (observation, reward)
        for turns in range(environment.number_of_turns):
            action = agent.calculate_action(percept)
            percept = environment.calculate_percept(action)
            summed_reward += Utility.get_reward_from_bitstring(percept[1])
    return summed_reward / number_of_evaluations


def train():
    """Train trainable untrained agents"""
    print("Training...")
    agent_classes = get_agents()
    for idx in range(len(agent_classes)):
        print("%s %d/%d" % (agent_classes[idx].__name__, idx + 1, len(agent_classes)))
        if agent_classes[idx].is_trainable:
            train_agent(agent_classes[idx])
    # save set of trained class environment tuples
    pickle.dump(is_trained_set, is_trained_set_path.open("wb"))


def train_agent(agent_class: Type[Agent]):
    """Train agent in environments"""
    environment_classes = get_environments()
    for idx in range(len(environment_classes)):
        print("    Training on %s - (%d/%d)" % (environment_classes[idx].__name__, idx + 1, len(environment_classes)))
        if not (agent_class, environment_classes[idx]) in is_trained_set:
            train_agent_environment(agent_class, environment_classes[idx], "0")
            train_agent_environment(agent_class, environment_classes[idx], "1")
            is_trained_set.add((agent_class, environment_classes[idx]))


def train_agent_environment(agent_class: Type[Agent], environment_class: Type[Environment], sign_bit: str):
    """Train an agent for training_steps steps in an environment"""
    agent = agent_class(environment_class, sign_bit, 0)
    for training_step in range(training_steps):
        environment = environment_class(sign_bit)
        observation = "1" * environment.observation_length
        reward = sign_bit + "0" * (environment.reward_length - 1)
        percept = (observation, reward)
        for turns in range(environment.number_of_turns):
            action = agent.calculate_action(percept)
            percept = environment.calculate_percept(action)
            agent.train(percept[1])
        if training_step % step_size == 0 and training_step != 0:
            agent.save(sign_bit, training_step)
        agent.reset()


def complexity() -> List:
    """Return a dictionary which holds all environments and their complexity"""
    print("Calculate complexity...")
    complexity_dict = []
    for environment_class in get_environments():
        env = environment_class("0")
        complexity_dict.append({
            "name": environment_class.__name__,
            "complexity": Utility.environment_complexity(env, env.randomness)
        })
    return complexity_dict


def get_agents() -> List[type(Agent)]:
    """Initialize and return a list of agents"""
    return [
        RandomActions,
        Handcrafted,
        NNrelu,
        NNrelu4,
        NNsigmoid,
        NNtanh
    ]


def get_environments() -> List[type(Environment)]:
    """Initialize and return a list of environments"""
    return [
        BiasedCoinFlip,
        DoubleCoinFlip,
        Slide,
        AnyOne,
        SpecificOne
    ]
