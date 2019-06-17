"""
Data histograms and model fit
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_distances(d, threshold=None, file=None):
    """
    Plot maximum distribution distances between time points for each gene.
    """
    T, G = d.shape
    x = np.arange(G)
    plt.figure(figsize=(8,4), dpi=100)
    y = -np.sort(-np.max(d, axis=0))
    plt.plot(x, y, linewidth=1.5, color='red', label='Distances')
    plt.xlim(0,G-1)
    plt.ylim(0,1)
    if threshold:
        y = threshold * np.ones(G)
        plt.plot(x, y, linewidth=1, color='black', ls='--', label='Threshold')
    plt.legend()
    if file is None: file = 'Distances'
    path = file + '.pdf'
    plt.savefig(path, dpi=100, bbox_inches='tight', frameon=False)
    plt.close()
