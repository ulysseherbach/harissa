"""
Main class for network inference and simulation
"""
import numpy as np
from scipy import sparse
from .cascade import cascade
from .tree import tree
from ..inference import inference, infer_kinetics
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
            self.a[2] = 0.02 # Constant Koff rate (normalized)
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
        # # Store time-dependent values
        # T = len(set(times))
        # self.va = np.zeros((T,G))
        # Kinetic values for each gene
        a = np.zeros((3,G))
        a[1,0], a[2,0] = 1, 1
        for g in range(1,G):
            if verb: print('Calibrating gene {}...'.format(g))
            x = data[:,g]
            at, b = infer_kinetics(x, times, verb=verb)
            a[1,g] = np.max(at)
            a[2,g] = b
            # self.va[:,g] = at
        self.a = a

    def fit(self, data, l=1, tol=1e-4, mask=None, sign=None, max_iter=1000,
        save=None, load=None, verb=False):
        """
        Fit the network model to the data.
        Return the list of successive objective function values.
        """
        C, G = data.shape
        times = list(set(data[:,0]))
        times.sort()

        # Default degradation rates
        d = np.zeros((2,G))
        d[0] = np.log(2)/9 # mRNA degradation rates
        d[1] = np.log(2)/46 # protein degradation rates
        self.d = d
        
        # Get kinetic parameters
        if load is None:
            self.get_kinetics(data, verb)
            if save is not None: np.save(save+'a', self.a)
            # if save is not None: np.save(save+'at', self.va)
        else: self.a = np.load(load+'a.npy')

        # Initialization
        x = data
        basal = np.zeros(G)
        inter = {t: np.zeros((G,G)) for t in times}
        a = self.a[1]
        b = self.a[2]
        c = 10 * np.ones(G)

        # Load a previous run
        if load is not None:
            basal = np.load(load+'basal.npy')
            inter_ = np.load(load+'inter.npy')
            inter = {t: inter_[k] for k, t in enumerate(times)}

        # Inference procedure
        y, q = inference(x, inter, basal, a, b, c, l, tol, mask, sign,
            max_iter, save, verb)

        # Build the results
        self.basal = basal
        self.inter_time = inter
        self.inter = np.zeros((G,G))
        for i in range(G):
            for j in range(G):
                val = np.array([inter[t][i,j] for t in set(times) - {0}])
                self.inter[i,j] = val[np.argmax(np.abs(val))]
        self.y = y
        self.q = q
        
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
        # Remove the sparse type if necessary
        if sparse.issparse(inter):
            inter = inter.toarray()
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
            # Remove the sparse type if necessary
            if sparse.issparse(inter):
                inter = inter.toarray()
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
        print(m)
        return Simulation(self.genes, t, m, p)

    def plot_obj(self, file=None):
        from harissa.graphics import plot_obj as plot
        plot(self.q, file=file)

    def plot_xy(self, data, g1=1, g2=2, time=True, file=None):
        from harissa.graphics import plot_xy as plot
        plot(data, self.y, g1=g1, g2=g2, time=time, file=file)

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
