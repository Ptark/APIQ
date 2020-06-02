import pickle
import numpy as np
import matplotlib.pyplot as plt

from python.src import Utility
from python.src.analysis import Colors

data_dir_path = Utility.get_data_path()
plots_path = Utility.get_plots_path()

apiq_dict_path = data_dir_path.joinpath("apiq_dict.apiq")
reward_dict_path = data_dir_path.joinpath("reward_dict.apiq")

apiq_dict = pickle.load(apiq_dict_path.open("rb"))
reward_dict = pickle.load(reward_dict_path.open("rb"))
reward_dict_sorted = {k: reward_dict[k] for k in apiq_dict.keys()}

# bar plot of apiq values
f = plt.figure()
xpos = [i for i in range(len(apiq_dict.keys()))]
plt.bar(xpos, list(apiq_dict.values()))
plt.xticks(xpos, list(apiq_dict.keys()))
plt.ylim(0, 1)
plt.show()
apiq_figure_path = plots_path.joinpath("apiq_figure.pdf")
f.savefig(apiq_figure_path, bbox_inches='tight')

#
environments = reward_dict_sorted[next(iter(reward_dict_sorted))]
colors = Colors.get_colors(len(environments), 1)
lighter_colors = Colors.get_colors(len(environments), 2)
positive_rewards = []
rewards = []
idx = 0
for env in environments.keys():
    positive_rewards.append([])
    rewards.append([])
    for agent in reward_dict_sorted:
        positive_reward = reward_dict_sorted[agent][env]["0"]
        negative_reward = reward_dict_sorted[agent][env]["1"]
        positive_rewards[idx].append(positive_reward)
        rewards[idx].append(positive_reward + negative_reward)
    idx += 1
width = 0.2
scale = (len(environments) + 4) * width
xpos = np.array([i * scale for i in range(len(reward_dict_sorted.keys()))])
for i in range(len(positive_rewards)):
    # plt.bar(xpos + width * i, positive_rewards[i], color=lighter_colors[i], width=width)
    plt.bar(xpos + width * i, rewards[i], color=colors[i], width=width)
plt.xticks(xpos + width * len(positive_rewards) / 2, list(apiq_dict.keys()))
plt.ylim(0, 1)
plt.show()

