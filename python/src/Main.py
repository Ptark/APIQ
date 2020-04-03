from python.src.agents.RandomAgent import RandomAgent
from python.src.environments.Slide import Slide
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip
from python.src.environments.DoubleCoinFlip import DoubleCoinFlip


def agents():
    return [RandomAgent()]


def environments():
    return [
        BiasedCoinFlip(),
        DoubleCoinFlip(),
        Slide(),
    ]


def evaluate_agent_in_environment(ag, env, num_eval, sign):
    total_reward = 0
    for i in range(num_eval):
        reward = 0
        percept = ((1, 1, 1, 1), (0, 0, 0, 0))
        for turns in range(env.turns):
            action = ag.calculate_action(percept)
            percept = env.calculate_percept(action)
            count = 0
            for x in percept[1]:
                reward += x * pow(0.5, count)
                count += 1
        total_reward += sign * reward
    return total_reward / number_of_evaluations


number_of_evaluations = 10000   # number of evaluations of the agent in an environment for stochastic purposes
agents = agents()      # list of agents
scores = []
for agent in agents:          # loop for evaluating all agents
    # initialize list of environments
    environments = environments()
    for environment in environments:                                # loop for evaluating agent in all environments
        scores.append(evaluate_agent_in_environment(agent, environment, number_of_evaluations, 1))
        scores.append(evaluate_agent_in_environment(agent, environment, number_of_evaluations, -1))
print(scores)




