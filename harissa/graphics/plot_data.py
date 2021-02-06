"""
Data histograms and model fit
"""
from ..inference.kinetics import infer_kinetics
import numpy as np
from numpy import exp, log
from scipy.special import gammaln
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import matplotlib.ticker as ticker

alpha = 1/2 # Transformation mRNA**alpha in plots

def estim_rep(X):
    """Estimate the repartition function of a sample."""
    l = np.size(X)
    x = np.append(np.append(0,np.sort(X)),1.2*np.max(X))
    y = np.append(np.linspace(0,1,l+1),1)
    return (x,y)

def distribution(x, a, b):
    """Negative binomial distribution with rate a and scale b."""
    if a > 0:
        c = (gammaln(x+a) - gammaln(x+1) - gammaln(a)
            + a*log(b) - (a+x)*log(b+1))
        return exp(c)
    elif a==0: return 1 * (x==0)

def prep_figure(ax):
    """Graphic options for histograms"""
    # ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(scilimits=(0, 0))
    ax.tick_params(labelsize = 10, left = False, right = False)
    ax.set_yticklabels([])

def plot_data(data, path=None, genes=None, ranks=False):
    """Plot the time-varying histogram of each gene."""
    if genes is None:
        G = data[0].size - 1
        genedict = {i+1: 'Gene {}'.format(i+1) for i in range(G)}
    else:
        G = genes[:,0].size
        genedict = {int(genes[i,0]): genes[i,1] for i in range(G)}
    # Time points
    t = np.sort(list(set(data[:,0])))
    T = t.size
    rank = 0
    # Plotting routine
    for idgene, gene in genedict.items():
        print('Plotting gene {} ({})...'.format(idgene, gene))
        rank += 1
        times = data[:,0]
        X = np.asarray(data[:,idgene], dtype='int')
        a, b = infer_kinetics(X, times)
        xmax = np.max(X)
        ### Set up the figure for the gene
        fig = plt.figure(figsize=(10,T*10/6))
        gs = gridspec.GridSpec(T,3)
        gs.update(hspace = 0.5)
        for i in range(T):
            ### Selection of the data for time t
            Xt = X[times==t[i]]
            ### Plot the mRNA molecule numbers
            ax = fig.add_subplot(gs[i,0])
            if (i == 0): ax.set_title(gene)
            hx = np.arange(0,xmax+1)
            hist = np.bincount(Xt, minlength=xmax+1)
            hy = hist/np.sum(hist)
            ax.plot(hx, hy, linewidth=1.5, color='lightgray')
            prep_figure(ax)
            ax.set_xlim(0,xmax)
            ax.set_ylim(0,1.05*np.max(hy))
            # Fitting negative binomial distributions
            x = np.arange(0,xmax+1)
            y = distribution(x, a[i], b)
            ax.plot(x, y, linewidth=1.5, color='red')
            ### Plot the transformed data mRNA**alpha
            ax = fig.add_subplot(gs[i,1])
            lp = r'$\mathregular{{^{{{:.1f}}}}}$'.format(alpha)
            if (i == 0): ax.set_title(gene+lp)
            ax.plot(hx**alpha, hy, linewidth=1.5, color='lightgray')
            prep_figure(ax)
            ax.set_xlim(0,xmax**alpha)
            ax.set_ylim(0,1.05*np.max(hy))
            ax.plot(x**alpha, y, linewidth=1.5, color='red', label='Model')
            ### Plot the repartition functions of mRNA**alpha
            ax = fig.add_subplot(gs[i,2])
            if (i == 0): ax.set_title('Rep. '+gene+lp)
            rx, ry = estim_rep(Xt**alpha)
            ax.step(rx, ry, linewidth=1.5, color='lightgray', label='Data')
            prep_figure(ax)
            ax.set_xlim(0,xmax**alpha)
            ax.set_ylim(0,1)
            rx = np.append(x**alpha, (xmax+1)**alpha)
            ry = np.append(y[0], np.cumsum(y))
            ax.step(rx, ry, linewidth=1.5, color='red', label='Model')
        if path is None: path = ''
        if ranks: file = path + 'Histo_{}_{}.pdf'.format(rank, idgene)
        else: file = path + 'Histo_{}.pdf'.format(idgene)
        fig.savefig(file, bbox_inches='tight', frameon=False)
        plt.close()
