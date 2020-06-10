import pickle
import numpy as np
import matplotlib.pyplot as plt
from python.src import Utility
from python.src.analysis import Colors
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

data_dir_path = Utility.get_data_path()
plots_path = Utility.get_plots_path()

reward_dict_path = data_dir_path.joinpath("reward_dict.apiq")
reward_dict = pickle.load(reward_dict_path.open("rb"))

#
environments = reward_dict[next(iter(reward_dict))]
colors = Colors.get_colors(len(environments), 1)
lighter_colors = Colors.get_colors(len(environments), 2)
positive_rewards = []
rewards = []
idx = 0
for env in environments.keys():
    positive_rewards.append([])
    rewards.append([])
    for agent in reward_dict:
        positive_reward = reward_dict[agent][env]["0"]
        negative_reward = reward_dict[agent][env]["1"]
        positive_rewards[idx].append(positive_reward)
        rewards[idx].append(positive_reward + negative_reward)
    idx += 1
f = plt.figure(figsize=(7, 4), dpi=400)
plt.grid()
width = 0.2
scale = (len(environments) + 4) * width
xpos = np.array([i * scale for i in range(len(reward_dict.keys()))])
for i in range(len(positive_rewards)):
    # plt.bar(xpos + width * i, positive_rewards[i], color=lighter_colors[i], width=width)
    plt.bar(xpos + width * i, rewards[i], color=colors[i], width=width)
plt.xticks(xpos + width * len(positive_rewards) / 2, list(reward_dict.keys()))
plt.ylim(0, 1)
reward_figure_path = plots_path.joinpath("reward_figure.pdf")
f.savefig(reward_figure_path, bbox_inches='tight')
plt.show()

