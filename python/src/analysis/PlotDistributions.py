import pickle
import matplotlib.pyplot as plt
import numpy as np
from python.src import Utility
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

data_dir_path = Utility.get_data_path()
plots_path = Utility.get_plots_path()

discrete_distribution_path = data_dir_path.joinpath("discrete_distribution.apiq")
continuous_distribution_path = data_dir_path.joinpath("continuous_distribution.apiq")

discrete_distribution = pickle.load(discrete_distribution_path.open("rb"))
continuous_distribution = pickle.load(continuous_distribution_path.open("rb"))

f = plt.figure(figsize=(7, 4), dpi=400)
x = np.arange(0, len(discrete_distribution))
plt.bar(x, discrete_distribution, color='#8ec07c')
plt.bar(x, continuous_distribution, color='#076678', alpha=.5)
plt.xlabel("Complexity in python bytecode instructions")
plt.ylabel("Density of environments")
plt.legend(labels=['Number of environments', 'Number of environments\nconvolved with normal\ndistribution'])
distributions_figure_path = plots_path.joinpath("distribution_figure.pdf")
f.savefig(distributions_figure_path, bbox_inches='tight')
plt.show()

