"""
Inferred protein levels
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_proteins(y, g1=1, g2=2, file=None):
    """
    Plot inferred protein levels conditionnaly to mRNA levels.
    """
    i, j = g1, g2
    fig = plt.figure(figsize=(5,5), dpi=100)
    plt.scatter(y[:,i], y[:,j], s=100, c='k', marker='.', edgecolors='none')
    ymin, ymax = np.min(y[:,[i,j]]), np.max(y[:,[i,j]])
    plt.xlim(ymin,ymax)
    plt.ylim(ymin,ymax)
    plt.axis('equal', 'box')
    if file is None: file = 'Proteins'
    path = file + '.pdf'
    fig.savefig(path, dpi=100, bbox_inches='tight', frameon=False)
    plt.close()
    
def plot_xy(x, y, g1=1, g2=2, file=None):
    """
    Plot inferred protein levels conditionnaly to mRNA levels.
    """
    i, j = g1, g2
    t = x[:,0]
    time = list(set(t))
    time.sort()
    T = len(time)
    cmap = plt.get_cmap('plasma', T+1)
    fig = plt.figure(figsize=(12,4), dpi=100)
    gs = gridspec.GridSpec(1,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    for k in range(T):
        cells = (t == time[k])
        # Observed mRNA levels
        ax1.scatter(x[cells,i], x[cells,j], s=50, c=[cmap(k)], marker='.',
            edgecolors='none', label='t = {}'.format(time[k]))
        # Inferred protein levels
        ax2.scatter(y[cells,i], y[cells,j], s=50, c=[cmap(k)], marker='.',
            edgecolors='none', label='t = {}'.format(time[k]))
    xmin, xmax = np.min(x[:,[i,j]]), np.max(x[:,[i,j]])
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(xmin,xmax)
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('Gene {}'.format(g1))
    ax1.set_ylabel('Gene {}'.format(g2))
    ax1.set_title('mRNA levels (data)')
    ax1.legend()
    ymax = np.max(y[:,[i,j]])
    ax2.set_xlim(0,np.max([1,ymax]))
    ax2.set_ylim(0,np.max([1,ymax]))
    ax2.set_aspect('equal', 'box')
    ax2.set_xlabel('Gene {}'.format(g1))
    ax2.set_ylabel('Gene {}'.format(g2))
    ax2.set_title('Protein levels (inferred)')
    ax2.legend()
    # Save figure
    if file is None: file = 'XY'
    path = file + '.pdf'
    fig.savefig(path, dpi=100, bbox_inches='tight', frameon=False)
    plt.close()
