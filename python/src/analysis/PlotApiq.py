import pickle
import matplotlib.pyplot as plt
from python.src import Utility
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

data_dir_path = Utility.get_data_path()
plots_path = Utility.get_plots_path()

apiq_dict_path = data_dir_path.joinpath("apiq_dict.apiq")
apiq_dict = pickle.load(apiq_dict_path.open("rb"))

# bar plot of apiq values
f = plt.figure(figsize=(7, 4), dpi=400)
colors = [
    ("#076678", 1),
    ("#076678", 0.75),
    ("#076678", 0.5),
    ("#076678", 0.25),
    ("#689d6a", 1),
    ("#8f3f71", 1),
    ("#8f3f71", 0.75),
    ("#8f3f71", 0.5),
    ("#8f3f71", 0.25),
]
agents = list(apiq_dict.keys())
apiq_values = list(apiq_dict.values())
ypos = [y for y in range(len(agents))]
for y in ypos:
    y_reverse = len(ypos) - 1 - y
    plt.text(apiq_values[y] + 3, y + .25, str(apiq_values[y]), color='black', fontweight='bold')
    plt.barh([y_reverse], [list(apiq_dict.values())[y]], color=colors[y][0], alpha=colors[y][1])
plt.yticks(ypos, reversed(list(apiq_dict.keys())), )
plt.xlabel("APIQ")
plt.xlim(0, 1)
apiq_figure_path = plots_path.joinpath("apiq_figure.pdf")
f.savefig(apiq_figure_path, bbox_inches='tight')
plt.show()
