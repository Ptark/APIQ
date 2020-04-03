from python.src.agents.RandomAgent import RandomAgent
from python.src.environments.BiasedCoinFlip import BiasedCoinFlip
from python.src.environments.DoubleCoinFlip import DoubleCoinFlip

number_of_evaluations = 1000   # number of evaluations of the agent in an environment for stochastic purposes
agents = [RandomAgent()]      # list of agents
scores = []
for agent in agents:          # loop for evaluating all agents
    environments = [(BiasedCoinFlip(), 1), (DoubleCoinFlip(), 1)]   # initialize list of environments
    for environment in environments:                                # loop for evaluating agent in all environments
        total_reward = 0
        for i in range(number_of_evaluations):
            percept = ((1, 1, 1, 1), (0, 0, 0, 0))
            for turns in range(environment[1]):
                action = agent.calculate_action(percept)
                percept = environment[0].calculate_percept(action)
                reward = percept[1][0] * (-1) + percept[1][1] * 1 + percept[1][2] * 0.5 + percept[1][3] * 0.25
                total_reward += reward
        scores.append(total_reward/number_of_evaluations)
    print(type(agent))
    print(scores)




