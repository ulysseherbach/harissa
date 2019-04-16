"""Monitoring and visualization of the results"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from .utils import estimRep, funRep

### Define colors
bleu = "#0048EC"
gris = "#F3F3F3"
rouge = "#EE0000"
vert = "#00CC00"
orange = "#FF6600"

### Local options
alpha = 1/2 # Transformation mRNA**alpha in plots

def prepFigure(ax):
    """Graphic options for histograms"""
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(scilimits=(0, 0))
    ax.tick_params(labelsize = 10, left = False, right = False)
    ax.set_yticklabels([])

def plotHistoGenes(Data, image_path, *lgenes, **models):
    """Plot the histogram of each gene."""
    nbins = 20 # Number of bins
    genedict = Data.genes(*lgenes)
    Timepoints = Data.timepoints()
    T = len(Timepoints) # Number of time-points
    D = Data.array # Retrieve the structured array
    for idgene, gene in genedict.items():
        print("Plotting gene {} ({})...".format(idgene,gene))
        X = D[gene]
        xmax = np.max(X)
        ### Set up the figure for the gene
        fig = plt.figure(figsize=(10,T*10/6), dpi=100)
        gs = gridspec.GridSpec(T,3)
        gs.update(hspace = 0.5)
        for t, timepoint in enumerate(Timepoints):
            ### Selection of the data for time t
            Xt = X[D['timepoint']==timepoint]
            ### Remove the UDs (coded by -1)
            Xt = Xt[Xt>=0]
            ### Plot the mRNA molecule numbers
            ax = fig.add_subplot(gs[t,0])
            if (t == 0): ax.set_title(gene)
            bins = np.linspace(0,xmax,nbins)
            n, bins, patches = ax.hist(Xt, bins, density=True, facecolor=gris, edgecolor='black')
            prepFigure(ax)
            ax.set_xlim(0,1.05*xmax)
            ax.set_ylim(0,1.05*np.max(n))
            if gene in models:
                x = np.linspace(0.001,xmax,200)
                y = models[gene].getDistribution(x,timepoint)
                ax.plot(x,y, linewidth=1.5, color=rouge)
                if (t == 0):
                    m = models[gene].a[1] # Get the Hill power value
                    ax.legend([r'$\mathregular{m=}$'+'{:.1f}'.format(m)],loc='upper right')
            ### Plot the transformed data mRNA**alpha
            ax = fig.add_subplot(gs[t,1])
            if (t == 0): ax.set_title(gene+r"$\mathregular{{^{{{:.1f}}}}}$".format(alpha))
            bins = np.linspace(0,xmax**alpha,nbins)
            n, bins, patches = ax.hist(Xt**alpha, bins, density=True, facecolor=gris, edgecolor='black', label="Data")
            prepFigure(ax)
            ax.set_xlim(0,1.05*xmax**alpha)
            ax.set_ylim(0,1.05*np.max(n))
            if gene in models:
                x = np.linspace(0.001,xmax**alpha,200)
                y = (1/alpha)*(x**(1/alpha - 1))*(models[gene].getDistribution(x**(1/alpha),timepoint))
                ax.plot(x,y, linewidth=1.5, color=rouge, label="Model")
            ### Plot the repartition functions of mRNA**alpha
            ax = fig.add_subplot(gs[t,2])
            if (t == 0): ax.set_title("Rep. "+gene+r"$\mathregular{{^{{{:.1f}}}}}$".format(alpha))
            x, y = estimRep(Xt**alpha)
            ax.step(x,y, linewidth=1, color="black", label="Data")
            prepFigure(ax)
            ax.set_xlim(0,xmax**alpha)
            ax.set_ylim(0,1)
            if gene in models:
                x = np.linspace(0.0001,xmax**alpha,10000)
                y = (1/alpha)*(x**(1/alpha - 1))*(models[gene].getDistribution(x**(1/alpha),timepoint))
                y = funRep(x,y)
                ax.plot(x,y/y[-1], linewidth=1.5, color=rouge, label="Model")
        path = image_path + "/Histo_{}.pdf".format(idgene)
        fig.savefig(path, dpi=100, bbox_inches = "tight", frameon = False)
        plt.close()

def plotInference(Va,Vtheta,Vl,c,gene,pathimage):
    """Monitor the inference process"""
    fig = plt.figure(figsize=(8,10), dpi=100)
    gs = gridspec.GridSpec(4,1, hspace=0.5)
    x = np.array(range(0,np.size(Va[:,0])))
    ### Define the figure
    ax = fig.add_subplot(gs[0])
    ax.plot(x, Vl, linewidth = 1.5, color=rouge, label="logL")
    ax.legend(loc='lower right')
    ax.set_title(gene)
    ### Evolution of k0 and k1
    ax = fig.add_subplot(gs[1])
    ax.plot(x, Va[:,0], linewidth = 1.5, color=orange, label=r'$\mathregular{k_0}$')
    ax.plot(x, c*Va[:,1]+Va[:,0], linewidth = 1.5, color=bleu, label=r'$\mathregular{k_1}$')
    ax.legend(ncol=2)
    ### Evolution of koff
    ax = fig.add_subplot(gs[2])
    ax.plot(x, Va[:,2], linewidth = 1.5, color=vert, label=r'$\mathregular{k_{off}/s}$')
    ax.legend()
    ### Evolution of theta
    ax = fig.add_subplot(gs[3])
    for t in range(0,np.size(Vtheta[0,:])):
        # labtheta = r'$\mathregular{{\theta^{{({:.0f})}}}}$'.format(Timepoints[t])
        ax.plot(x, Vtheta[:,t], linewidth = 1.5)
    ax.legend([r'$\mathregular{\theta^{(t)}}$'],loc='upper left')
    fig.savefig(pathimage+".pdf", dpi=100, bbox_inches = "tight", frameon = False)
    plt.close()

def plotPosterior(dataset, posterior, image_path, *lgenes):
    """Plot the results of the inference, i.e. the posterior p(c|data)."""
    genedict = dataset.genes(*lgenes)
    for idgene, gene in genedict.items():
        print("Plotting posterior for gene {} ({})...".format(idgene,gene))
        P = posterior[posterior['idgene'] == idgene]
        Vc = P['c']
        Vp = P['p']
        Vm = P['a'][:,1]
        ### Define the figure
        fig = plt.figure(figsize=(8,6), dpi=100)
        gs = gridspec.GridSpec(2,1)
        # gs.update(hspace = 0.5)
        ### Pseudo-posterior of c
        ax = fig.add_subplot(gs[0])
        ax.plot(Vc, Vp, linewidth = 1.5, marker='o', color="orange", label=r'$\mathregular{p(c\mid x)}$')
        ax.set_xticks(Vc)
        ax.set_ylim(0,1.05*np.max(Vp))
        ax.set_title(gene)
        ax.legend()
        ### Pseudo-posterior of m
        ax = fig.add_subplot(gs[1])
        ax.plot(Vm, Vp, linewidth = 1.5, marker='o', color="red", label=r'$\mathregular{p(m\mid x)}$')
        ax.set_ylim(0,1.05*np.max(Vp))
        ax.legend()
        ### Export the plot
        path = image_path + "/Posterior_{}.pdf".format(idgene)
        fig.savefig(path, dpi=100, bbox_inches = "tight", frameon = False)
        plt.close()

def plotHistoHill(file,modeldict,nbins=None):
    """Plot the histogram of Hill powers from an inference output."""
    m = [model.a[1] for gene, model in modeldict.items()]
    fig = plt.figure(figsize=(6,4), dpi=100)
    gs = gridspec.GridSpec(1,1)
    ax1 = fig.add_subplot(gs[0])
    n, bins, patches = ax1.hist(m, bins=nbins, density=True, label="m")
    ax1.legend()
    fig.savefig(file, dpi=100, bbox_inches = "tight", frameon = False)
    plt.close()