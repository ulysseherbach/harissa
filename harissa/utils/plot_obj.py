"""
Objective function values along the inference algorithm
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_obj(q, file=None):
    """
    Plot values of objective function along the inference algorithm.
    """
    x = np.arange(len(q[1:]))
    y = np.array(q[1:])
    plt.plot(x, y, linewidth=1.5, color='red', label='Objective')
    plt.xlim(1,np.max(x))
    plt.ylim(np.min(y),np.max(y))
    plt.legend()
    if file is None: file = 'Objective'
    path = file
    plt.savefig(path, dpi=100, bbox_inches='tight', frameon=False)
    plt.close()
    