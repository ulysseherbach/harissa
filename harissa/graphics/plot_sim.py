"""
Plot trajectories
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_sim(t, m, p, genes, file=None):
    """
    Plot the expression path of a gene network
    """
    G = len(genes)
    cmap = plt.get_cmap('tab10') # Get the default color cycle
    fig = plt.figure(figsize=(12,4.75), dpi=100)
    gs = gridspec.GridSpec(2,1)
    gs.update(hspace=0.55)
    # gs.update(left=0.1, right=0.5, wspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.set_title('mRNA')
    ax2.set_title('Proteins')
    for i in range(G):
        g = genes[i]
        ax1.plot(t, m[:,i], c=cmap(i), lw=1.5, label='Gene {}'.format(g))
        ax1.set_xlim(t[0], t[-1])
        ax1.set_ylim(0, 1*np.max(m))
        ax1.legend(loc='upper right')
        ax2.plot(t, p[:,i], c=cmap(i), lw=1.5, label='Gene {}'.format(g))
        ax2.set_xlim(t[0], t[-1])
        ax2.set_ylim(0, np.max([1*np.max(p), 1]))
        ax2.legend(loc='upper right')
    # Save figure
    if file is None: file = 'Traj.pdf'
    fig.savefig(file, dpi=100, bbox_inches='tight', frameon=False)
    plt.close()
