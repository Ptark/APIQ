import pickle
import matplotlib.pyplot as plt
from python.src import Utility
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)
plt.rcParams['axes.axisbelow'] = True

data_dir_path = Utility.get_data_path()
plots_path = Utility.get_plots_path()

apiq_dict_path = data_dir_path.joinpath("apiq_dict.apiq")
apiq_dict = pickle.load(apiq_dict_path.open("rb"))

# bar plot of apiq values
f = plt.figure(figsize=(6, 4), dpi=400)
agents = list(apiq_dict.keys())
apiq_values = [v["mean"] for v in apiq_dict.values()]
apiq_errors = [v["error"] for v in apiq_dict.values()]
ypos = [len(agents) - 1 - y for y in range(len(agents))]
plt.barh(ypos[:4], apiq_values[:4], color="#076678")
plt.barh(ypos[4:5], apiq_values[4:5], color="#689d6a")
plt.barh(ypos[5:], apiq_values[5:], color="#8f3f71")
plt.errorbar(apiq_values, ypos, xerr=apiq_errors, fmt=',', ecolor='black', capsize=4)
plt.yticks(ypos, list(apiq_dict.keys()), )
plt.xlabel("APIQ")
plt.xlim(0, 1)
plt.grid()
plt.legend(["PiAgents", "Handcrafted", "NNAgents", "Error bars"])
apiq_figure_path = plots_path.joinpath("apiq_figure.pdf")
f.savefig(apiq_figure_path, bbox_inches='tight')
plt.show()
