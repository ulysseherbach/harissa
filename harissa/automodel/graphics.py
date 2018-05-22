"""Monitoring and visualization of the results"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .utils import estim_dist

### Define colors
bleu = "#0048EC"
gris = "#F3F3F3"
rouge = "#EE0000"
vert = "#00CC00"
orange = "#FF6600"

### Plot the path of theta along the EM algorithm
def plot_em(name, traj, size):
    G = size
    vq, vtheta = traj
    iter_em = np.size(vq)
    x = np.array(range(iter_em))
    e = 0 if vtheta is None else 1
    fig = plt.figure(figsize=(10,3 + 3*e), dpi=100)
    gs = gridspec.GridSpec(2 + e, 1)
    ax0 = fig.add_subplot(gs[0])
    label = r'$\mathregular{\widehat{L}}$'
    ax0.plot(x, vq, linewidth=1.5, color=rouge, label=label)
    # ax0.plot(x, vq2, linewidth=1.5, color=vert, label='L')
    ax0.legend(loc='upper left')
    if vtheta is not None:
        ax1 = fig.add_subplot(gs[2])
        ax2 = fig.add_subplot(gs[1])
        for i in range(G):
                y = vtheta['{}-{}'.format(i+1,i+1)]
                style = '--'
                ax2.plot(x, y, linewidth=1.5, ls=style, alpha=0.5)
                for j in range(i+1,G):
                    inter = '{}-{}'.format(i+1,j+1)
                    y = vtheta[inter]
                    label = r'$\mathregular{\theta_{'
                    label += '{},{}'.format(i+1,j+1)
                    label += r'}}$'
                    if np.abs(y[-1]) < 0.1: label = None
                    alpha = 0.5 if (i == j) else 1
                    style = '--' if (i == j) else '-'
                    ax1.plot(x, y, linewidth=1.5,
                        ls=style, alpha=alpha, label=label)
        if (G <= 10): ax1.legend(loc='upper left')
    fig.savefig(name, dpi=100, bbox_inches = "tight", frameon = False)
    plt.close()

### Plot the marginals of mRNA and modes
def plot_marginals(name, x, z, c):
    G = np.size(x[0,:])
    fig = plt.figure(figsize=(10,int(2*G)), dpi=100)
    gs = gridspec.GridSpec(G, 2, hspace=0.5)
    vmax = np.max(x)
    for i in range(G):
        ax0 = fig.add_subplot(gs[i,0]) # Modes
        ax1 = fig.add_subplot(gs[i,1]) # mRNAs
        # cmax = int(np.max(z[:,i]))
        p = estim_dist(z[:,i], c[i])
        xp = np.array(range(c[i]+1))
        ax0.plot(xp, p, linewidth = 1.5, marker = 'o', color=vert)
        ax0.set_xticks(xp)
        ax0.set_xlim(0,c[i])
        ax0.set_ylim(0,1)
        ax1.hist(x[:,i], bins=30, range=(0,vmax), facecolor=bleu)
        ax1.set_xlim(0, vmax)
        ax0.set_title(r'$\mathregular{Z_{'+str(i)+r'}}$')
        ax1.set_title(r'$\mathregular{X_{'+str(i)+r'}}$')
    fig.savefig(name, dpi=100, bbox_inches = "tight", frameon = False)
    plt.close()

### Plot the 2D histogram for samples of 2-gene networks
def plot_2d_histo(name, x, z):
    None