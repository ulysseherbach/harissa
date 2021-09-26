"""
Main class for network inference and simulation
"""
import numpy as np
from .cascade import cascade
from .tree import tree
from ..inference import infer_kinetics, infer_proteins, infer_network
from ..simulation import Simulation, BurstyPDMP, ApproxODE

class NetworkModel:
    """
    Handle networks within Harissa.
    """
    def __init__(self, n_genes=None):
        # Kinetic parameters
        self.a = None
        self.d = None
        # Network parameters
        self.basal = None
        self.inter = None
        # Default behaviour
        if n_genes is not None:
            G = n_genes + 1 # Genes plus stimulus
            # Default bursting parameters
            self.a = np.zeros((3,G))
            self.a[0] = 0 # Minimal Kon rate (normalized)
            self.a[1] = 2 # Maximal Kon rate (normalized)
            self.a[2] = 0.02 # Inverse burst size of mRNA
            # Default degradation rates
            self.d = np.zeros((2,G))
            self.d[0] = np.log(2)/9 # mRNA degradation rates
            self.d[1] = np.log(2)/46 # protein degradation rates
            # Default network parameters
            self.basal = np.zeros(G)
            self.inter = np.zeros((G,G))

    def get_kinetics(self, data, verb):
        """
        Compute the basal parameters of filtered genes.
        """
        times = data[:,0]
        G = data[0].size
        # Kinetic values for each gene
        a = np.ones((3,G))
        for g in range(1,G):
            if verb: print(f'Calibrating gene {g}...')
            x = data[:,g]
            at, b = infer_kinetics(x, times, verb=verb)
            a[0,g] = np.min(at)
            a[1,g] = np.max(at)
            a[2,g] = b
        self.a = a

    def fit(self, data, l=1, tol=1e-5, verb=False):
        """
        Fit the network model to the data.
        Return the list of successive objective function values.
        """
        x = data
        # Time points
        times = np.sort(list(set(x[:,0])))
        self.times = times
        # Default degradation rates
        C, G = x.shape
        d = np.zeros((2,G))
        d[0] = np.log(2)/9 # mRNA degradation rates
        d[1] = np.log(2)/46 # protein degradation rates
        self.d = d
        # Kinetic parameters
        self.get_kinetics(data, verb)
        a = self.a
        # Concentration parameter
        c = 100 * np.ones(G)
        # Get protein levels
        y = infer_proteins(x, a)
        self.y = y
        # Inference procedure
        theta = infer_network(x, y, a, c, l, tol, verb)
        # Build the results
        self.basal = np.zeros(G)
        self.inter = np.zeros((G,G))
        self.basal_time = {time: np.zeros(G) for time in times}
        self.inter_time = {time: np.zeros((G,G)) for time in times}
        for t, time in enumerate(times):
            self.basal_time[time][:] = theta[t][:,0]
            self.inter_time[time][:,1:] = theta[t][:,1:]
        self.basal[:] = theta[-1][:,0]
        self.inter[:,1:] = theta[-1][:,1:]

    def simulate(self, t, burnin=None, genes=None, verb=False):
        """
        Perform simulation of the network model (bursty PDMP version).
        """
        # Check parameters
        test = ((self.a is None) + (self.d is None)
                + (self.basal is None) + (self.inter is None))
        # Prepare time points
        if test: raise ValueError('Model parameters not yet specified')
        if np.size(t) == 1: t = np.array([t])
        if np.any(t != np.sort(t)):
            raise ValueError('Time points must appear in increasing order')
        a = self.a
        d = self.d
        basal = self.basal
        inter = self.inter
        network = BurstyPDMP(a, d, basal, inter)
        # Burnin simulation without stimulus
        if burnin is not None: network.simulation([burnin], verb)
        # Activate the stimulus
        network.state['P'][0] = 1
        # Final simulation with stimulus
        sim = network.simulation(t, verb)
        m, p = sim['M'], sim['P']
        return Simulation(t, m, p)

    def simulate_ode(self, t, burnin=None, verb=False):
        """
        Perform simulation of the network model (ODE version).
        """
        # Check parameters
        test = ((self.a is None) + (self.d is None)
                + (self.basal is None) + (self.inter is None))
        # Prepare time points
        if test: raise ValueError('Model parameters not specified yet')
        if self.inter is None: print('Interactions must be specified')
        if np.size(t) == 1: t = np.array([t])
        if np.any(t != np.sort(t)):
            raise ValueError('Time points must appear in increasing order')
        if self.genes is None:
            raise ValueError('genes not yet provided')
        v = [0] + self.genes
        # Case 1 (no filtering): all genes are simulated
        if self.filter is None:
            a = self.a
            d = self.d
            basal = self.basal
            inter = self.inter
        # Case 2 (filtering): only filtered genes are simulated
        else:
            G = len(v)
            a = np.zeros((3,G))
            d = np.zeros((2,G))
            basal = np.zeros(G)
            inter = np.zeros((G,G))
            for i in range(G):
                a[0,i] = self.a[0,v[i]]
                a[1,i] = self.a[1,v[i]]
                a[2,i] = self.a[2,v[i]]
                d[0,i] = self.d[0,v[i]]
                d[1,i] = self.d[1,v[i]]
                basal[i] = self.basal[v[i]]
                for j in range(G):
                    inter[i,j] = self.inter[v[i],v[j]]
        network = ApproxODE(a, d, basal, inter)
        # Burnin simulation without stimulus
        if burnin is not None:
            network.simulation([burnin], verb=verb)
        # Activate the stimulus
        network.state['P'][0] = 1
        # Final simulation with stimulus
        sim = network.simulation(t, verb=verb)
        m, p = sim['M'], sim['P']
        return Simulation(self.genes, t, m, p)

# Classes for simulations
class Cascade(NetworkModel):
    """
    Particular network with a cascade structure.
    """
    def __init__(self, n_genes, autoactiv=False):
        # Get NetworkModel default features
        NetworkModel.__init__(self, n_genes)
        # New network parameters
        basal, inter = cascade(n_genes)
        if autoactiv:
            for i in range(1,n_genes+1):
                inter[i,i] = 5
        self.basal = basal
        self.inter = inter

class Tree(NetworkModel):
    """
    Random network with a tree structure.
    """
    def __init__(self, n_genes, autoactiv=False):
        # Get NetworkModel default features
        NetworkModel.__init__(self, n_genes)
        # New network parameters
        basal, inter = tree(n_genes)
        if autoactiv:
            for i in range(1,n_genes+1):
                inter[i,i] = 5
        self.basal = basal
        self.inter = inter
