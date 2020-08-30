import pickle
import numpy as np
import matplotlib.pyplot as plt
from python.src import Utility
from python.src.analysis import Colors
from matplotlib import rc
from matplotlib import transforms
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

data_dir_path = Utility.get_data_path()
plots_path = Utility.get_plots_path()

reward_dict_path = data_dir_path.joinpath("reward_dict.apiq")
reward_dict = pickle.load(reward_dict_path.open("rb"))

#
environments = reward_dict[next(iter(reward_dict))]
colors = Colors.get_colors(len(environments))
rewards = []
errors = []
no_of_agents = len(reward_dict)
idx = 0
for env in environments.keys():
    rewards.append([])
    errors.append([])
    for agent in reward_dict:
        positive_reward = reward_dict[agent][env]["positive"]["mean"]
        negative_reward = reward_dict[agent][env]["negative"]["mean"]
        p_err = reward_dict[agent][env]["positive"]["error"]
        n_err = reward_dict[agent][env]["negative"]["error"]
        rewards[idx].append(positive_reward + negative_reward)
        errors[idx].append(np.sqrt(p_err * p_err + n_err * n_err))
    idx += 1
height = 1
scale = (len(environments) + 2) * height

f = plt.figure(figsize=(7.8, 8.75), dpi=400)
ax = f.add_subplot(1, 1, 1)
no_pi = 5
ypos = np.array([(no_pi - 1 - i) * scale for i in range(no_pi)])
for i in range(len(rewards)):
    pos = ypos + height * len(rewards) - height * i
    plt.barh(pos, rewards[i][:no_pi], color=colors[i], height=height)
for i in range(len(errors)):
    pos = ypos + height * len(errors) - height * i
    plt.errorbar(rewards[i][:no_pi], pos, xerr=errors[i][:no_pi], fmt=',', ecolor='black',
                 elinewidth=0.6, capsize=2)
ax.set_ylim(0, ypos[0] + scale)
ax.set_xlim(-0.75, 1.1)
ax.set_yticks(ypos + scale / 2)
ax.set_yticklabels(labels=list(reward_dict.keys()))
plt.tick_params(axis='y', which='major', left=False)
ax.set_yticks(ypos[:-1], minor=True)
plt.setp(ax.yaxis.get_majorticklabels(), ha='left')
dx, dy = 1.0, 0.53
offset = transforms.ScaledTranslation(dx, dy, f.dpi_scale_trans)
for label in ax.yaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)
ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
ax.set_xticks([-0.5, 0.0, 0.5, 1.0], minor=True)
ax.set_xlabel("Measured mean reward in environment and negation")
ax.grid(which='minor', color='black')
ax.xaxis.grid(True, which='major')
legend = [key for key in environments.keys()] + ["Error bars"]
ax.legend(legend, loc='center',
          bbox_to_anchor=(0.5, 1.1),
          ncol=3, fontsize=10.5)
reward_figure_path = plots_path.joinpath("reward_figure.pdf")
f.savefig(reward_figure_path, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------
f = plt.figure(figsize=(7.8, 10.5), dpi=400)
ax = f.add_subplot(1, 1, 1)
no_nn = 6
ypos = np.array([(no_nn - 1 - i) * scale for i in range(no_nn)])
for i in range(len(rewards)):
    pos = ypos + height * len(rewards) - height * i
    plt.barh(pos, rewards[i][no_pi:], color=colors[i], height=height)
for i in range(len(errors)):
    pos = ypos + height * len(errors) - height * i
    plt.errorbar(rewards[i][no_pi:], pos, xerr=errors[i][no_pi:], fmt=',', ecolor='black',
                 elinewidth=0.6, capsize=2)
ax.set_ylim(0, ypos[0] + scale)
ax.set_xlim(-0.75, 1.1)
ax.set_yticks(ypos + scale / 2)
ax.set_yticklabels(labels=list(reward_dict.keys())[no_pi:])
plt.tick_params(axis='y', which='major', left=False)
ax.set_yticks(ypos[:-1], minor=True)
plt.setp(ax.yaxis.get_majorticklabels(), ha='left')
offset = transforms.ScaledTranslation(dx, dy, f.dpi_scale_trans)
for label in ax.yaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)
ax.set_xticks([-0.5, 0.0, 0.5, 1.0])
ax.set_xticks([-0.5, 0.0, 0.5, 1.0], minor=True)
ax.set_xlabel("Measured mean reward in environment and negation")
ax.grid(which='minor', color='black')
ax.xaxis.grid(True, which='major')
reward_figure_path = plots_path.joinpath("reward_figure2.pdf")
f.savefig(reward_figure_path, bbox_inches='tight')
plt.show()
