"""Plotting utilities for the grnsim package"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

bleu = "#0048EC"
orange = "#FF6600"

def traceProm(T,E,ax,y,h,couleur):
    """Plot the promoter active periods"""
    start = []
    end = []
    state = False
    ax.set_xlim(np.min(T), np.max(T))
    ax.tick_params(left=False)
    ax.set_yticklabels([])
    for i in range(0,np.size(T)):
        if ((E[i] == 1) & (state == False)):
            start.append(T[i])
            state = True
        if ((E[i] == 0) & (state == True)):
            end.append(T[i])
            state = False
    if (state == True):
        end.append(T[-1])
    for x0, x1 in zip(start,end):
        ax.axvspan(x0, x1, ymin=y - h/2, ymax=y + h/2, color=couleur)

def plotsim(timepoints, expression, fname=False, **kwargs):
    """Plot the expression path of a gene network"""
    # fname = kwargs.get('fname', None)
    q0M, q1M = kwargs.get('q0M', None), kwargs.get('q1M', None)
    q0P, q1P = kwargs.get('q0P', None), kwargs.get('q1P', None)
    G = np.size(expression[0])
    model = 'full' if (len(expression[0].dtype) == 3) else 'bursty'
    cmap = plt.get_cmap("tab10") # Get the default color cycle
    if (model == 'full'):
        fig = plt.figure(figsize=(12,6), dpi=100)
        gs = gridspec.GridSpec(3,1,height_ratios=[0.5,1,1])
        gs.update(hspace=0.55)
        # gs.update(left=0.1, right=0.5, wspace=0.4)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        # ax0.set_yticks([(i+0.5)/G for i in range(G)])
        ax0.set_title('Promoter active periods')
        ax1.set_title('mRNA')
        ax2.set_title('Proteins')
        for i in range(G):
            ### Plot promoters
            traceProm(timepoints, expression['E'][:,i],
                ax0, (G-1-i+0.5)/G, 0.9/G, cmap(i))
            ### Plot mRNA
            ax1.plot(timepoints, expression['M'][:,i],
                linewidth=1.5, label='Gene {}'.format(i+1), color=cmap(i))
            ### Plot proteins
            ax2.plot(timepoints, expression['P'][:,i],
                linewidth=1.5, label='Gene {}'.format(i+1), color=cmap(i))
            ax2.legend(loc='upper left')
    elif (model == 'bursty'):
        fig = plt.figure(figsize=(12,6), dpi=100)
        gs = gridspec.GridSpec(2,1)
        gs.update(hspace=0.55)
        # gs.update(left=0.1, right=0.5, wspace=0.4)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax1.set_title('mRNA')
        ax2.set_title('Proteins')
        for i in range(G):
            ### mRNA quantiles
            if q0M is not None:
                ax1.plot(timepoints, q0M[:,i], c=cmap(i), alpha=0.25, lw=1.5)
            if q1M is not None:
                ax1.plot(timepoints, q1M[:,i], c=cmap(i), alpha=0.25, lw=1.5)
            ### plot mRNA
            ax1.plot(timepoints, expression['M'][:,i], c=cmap(i), lw=1.5,
                label='Gene {}'.format(i+1))
            ### Protein quantiles
            if q0P is not None:
                ax2.plot(timepoints, q0P[:,i], c=cmap(i), alpha=0.25, lw=1.5)
            if q1P is not None:
                ax2.plot(timepoints, q1P[:,i], c=cmap(i), alpha=0.25, lw=1.5)
            ### Plot proteins
            ax2.plot(timepoints, expression['P'][:,i], c=cmap(i), lw=1.5,
                label='Gene {}'.format(i+1))
            ax2.legend(loc='upper left')
    ### Save figure
    if fname is not None:
        fig.savefig(fname, dpi=100, bbox_inches='tight', frameon=False)
        plt.close()

def histo(M, P, fname=None):
    """Display the marginal distributions"""
    G = np.size(M[0,:])
    fig = plt.figure(figsize=(10,int(2*G)), dpi=100)
    gs = gridspec.GridSpec(G, 2, hspace=0.5)
    Mmax, Pmax = np.max(M), np.max(P)
    cmap = plt.get_cmap("tab10") # Get the default color cycle
    for i in range(G):
        ax0 = fig.add_subplot(gs[i,0]) # mRNAs
        ax1 = fig.add_subplot(gs[i,1]) # Proteins
        ax0.hist(M[:,i], bins=30, range=(0,Mmax), facecolor=cmap(i))
        ax0.set_xlim(0, Mmax)
        ax1.hist(P[:,i], bins=30, range=(0,Pmax), facecolor=cmap(i))
        ax1.set_xlim(0, Pmax)
        ax0.set_title(r'$\mathregular{M_{'+str(i+1)+r'}}$')
        ax1.set_title(r'$\mathregular{P_{'+str(i+1)+r'}}$')
    ### Save figure
    if fname is not None:
        fig.savefig(fname, dpi=100, bbox_inches='tight', frameon=False)
        plt.close()