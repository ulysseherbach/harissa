# Example of network layout using harissa.utils
import numpy as np
import matplotlib.pyplot as plt
from harissa.utils import build_pos, plot_network

# Path of result files
result_path = "../examples/results/"

# Interaction matrix
inter = np.array([
    [ 1, 1, 1],
    [-1,-1,-1],
    [ 1,-1, 1]])

# Node labels and positions
names = [f"$G_{{{i+1}}}$" for i in range(inter[0].size)]
# Option 1: layout from networkx
pos = build_pos(inter)
# Option 2: layout from networkx + graphviz (needs installation)
# pos = build_pos(inter, method='graphviz')

# Figure
fig = plt.figure(figsize=(5,5))
ax = fig.gca()

# Draw the network
plot_network(inter, pos, axes=ax, names=names, scale=2)

# Export the figure
fig.savefig(result_path + "test_plot_network.pdf", bbox_inches='tight')
