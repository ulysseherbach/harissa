"""
Main class for network inference and simulation
"""
import numpy as np
from scipy import sparse
from ..generator import cascade, tree
from ..inference import (distance_matrix, network_filter, infer_kinetics,
    genes_best, variation_matrix, network_filter_mechanistic, inference)
from harissa.simulation import Simulation, BurstyPDMP, ApproxODE

########### A FAIRE #############
# Ne pas transférer le namespace des modules
# Enlever les None par défaut et remplacer ça par :
# try:
#     doStuff(a.property)
# except AttributeError:
#     otherStuff()
#################################

class NetworkModel:
    """
    Handle networks within Harissa.

    There are three intermediary results that can be saved:
    - distances : matrix of distances (see filtering.py)
    - filter : sparse matrix with filtered interactions
    - basal : basal parameters of filtered genes

    genes : None or list of positive integers
        If None, all genes are considered (as many as basal rows)
        If list, it corresponds to the subset of filtered relevant genes
    """
    def __init__(self, n_genes=None):
        # Auxiliary parameters
        self.distances = None
        self.variations = None
        self.filter = None
        self.genes = None
        # Kinetic parameters
        self.a = None
        self.d = None
        self.a_time = None
        # Network parameters
        self.basal = None
        self.inter = None
        # Default behaviour
        if n_genes is not None:
            G = n_genes + 1 # Genes plus stimulus
            self.genes = list(range(1,G))
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
            self.inter = sparse.dok_matrix((G,G))

    def get_distances(self, data, file=None, verb=False):
        """
        Compute the matrix of distribution distances between time points.
        """
        d = distance_matrix(data, verb=verb)
        if file is not None: np.save(file+'_distances', d)
        self.distances = d

    def get_filter(self, threshold=0, alpha=None, l1=0.5,
        file=None, verb=False):
        """
        Compute the sparse matrix of filtered interactions.
        """
        if self.distances is None:
            raise ValueError('distances not yet provided')
        if self.filter is None:
            f = network_filter(self.distances, threshold, alpha, l1, verb)
            if file is not None: sparse.save_npz(file+'_filter', f)
            self.filter = f

    def get_kinetics(self, data, threshold=0, file=None, verb=False):
        """
        Compute the basal parameters of filtered genes.
        """
        t = np.sort(list(set(data[:,0])))
        G, T = data[0].size, t.size
        if (self.filter is None) and (threshold == 0):
            # Dense matrices
            a = np.zeros((3,G))
            d = np.zeros((2,G))
            a_time = np.zeros((T,G))
            # Full gene list
            self.genes = list(range(1,G))
        else:
            # Sparse matrices
            a = sparse.lil_matrix((3,G))
            d = sparse.lil_matrix((2,G))
            a_time = sparse.lil_matrix((T,G))
            t = np.reshape(t, (T,1))
            if self.filter is None:
                # Gene list in case of nonzero threshold
                self.genes = genes_best(data, threshold, verb=verb)
            else:
                # Gene list in case of pre-computed filter
                li, lj = self.filter.nonzero()
                genes = set(li).union(lj)
                genes.discard(0)
                self.genes = list(genes)
        # Get values for filtered genes
        for g in self.genes:
            # if verb: print('Estimating kinetics of gene {}'.format(g))
            x = data[:,g]
            times = data[:,0]
            at, b = infer_kinetics(x, times)
            # a[0,g] = np.min(at)
            a[1,g] = np.max(at)
            a[2,g] = b
            # if self.d is None:
            #     d[0,g] = np.log(2)/9 
            #     d[1,g] = np.log(2)/46
            if (self.filter is None) and (threshold == 0): a_time[:,g] = at
            else: a_time[:,g] = np.reshape(at, (T,1))
        # Default values for the stimulus (ignored)
        a[1,0] = 1
        a[2,0] = 1
        # d[0,0] = 1
        # d[1,0] = 0.2
        a_time[:,0] = t # Store the list of time points
        # # OPTION: set a unique a[1] for all genes
        # a1 = np.max(a[1,1:].toarray())
        # for g in self.genes: a[1,g] = a1
        # Save the results
        if file is not None:
            if (self.filter is None) and (threshold == 0):
                np.save(file+'_a', a)
                np.save(file+'_d', d)
                np.save(file+'_a_time', a_time)
            else:
                sparse.save_npz(file+'_a', a.asformat('csr'))
                sparse.save_npz(file+'_d', d.asformat('csr'))
                sparse.save_npz(file+'_a_time', a_time.asformat('csr'))
        self.a = a
        # self.d = d
        self.a_time = a_time

    def get_variations(self, file=None):
        """
        Mechanistic analog of `get_distances`.
        It has to be called after `get_kinetics`.
        """
        if self.a_time is None:
            raise ValueError('kinetic parameters not yet provided')
        # Here we reduce a_time into a minimal basic array
        genes = self.genes
        T, G = self.a_time.shape
        G0 = len(genes) + 1 # Filtered genes + stimulus
        t = np.array([self.a_time[i,0] for i in range(T)])
        a_time = np.zeros((T,G0))
        for i in range(T):
            for g in range(1,G0):
                a_time[i,g] = self.a_time[i,genes[g-1]]
        # Symbolize the stimulus
        a_stimulus = np.max(a_time[:,1:]) - np.min(a_time[:,1:])
        a_time[:,0] = a_stimulus
        a_time[0,0] = 0
        v = variation_matrix(a_time, t)
        if file is not None: np.save(file+'_variations', v)
        self.variations = v

    def get_filter_mechanistic(self, alpha=None, file=None, verb=False):
        """
        More mechanistic variant of `get_filter`.
        This one has to be called after `kinetic parameters`.
        """
        if self.a is None:
            raise ValueError('kinetic parameters not yet provided')
        T, G = self.a_time.shape
        genes = self.genes
        v = self.variations
        f = network_filter_mechanistic(G, v, genes, alpha=alpha, verb=verb)
        if file is not None: sparse.save_npz(file+'_filter_mechanistic', f)
        self.filter = f

    def fit(self, data, threshold=0, alpha=None, l=1e-2, tol=1e-4,
        max_iter=100, dense=True, sign=False, l1=0.5, verb=False):
        """
        Fit the network model to the data.
        Return the list of successive objective function values.
        """
        C, G = data.shape
        times = set(data[:,0])

        # Preprocessing
        self.get_distances(data, verb=verb)
        self.get_filter(threshold=threshold, alpha=alpha, l1=l1, verb=verb)
        self.get_kinetics(data, verb=verb)

        # OPTION: add the first moving gene
        g1 = np.argmax(self.distances[0,1:]) + 1
        if verb: print('First wave: gene {}'.format(g1))

        # Get the minimal list of genes
        if self.filter is not None:
            # Keep only interacting genes
            I, J = self.filter.nonzero()
            # genes = list(set(I) | set(J) | {0})
            genes = list(set(I) | set(J) | {0} | {g1})
            genes.sort()
        else: genes = list(range(G))
        G0 = len(genes)

        # Initialization
        x = data[:,genes]
        basal = np.zeros(G0)
        a = np.zeros(G0)
        b = np.zeros(G0)
        c = np.zeros(G0)
        for i, g in enumerate(genes):
            a[i] = self.a[1,g]
            b[i] = self.a[2,g]
            c[i] = 10

        # Build the interaction mask
        gene = {g: i for i, g in enumerate(genes)} # Inverted dictionary
        if self.filter is not None:
            m = np.zeros((G0,G0), dtype='int')
            I, J, V = sparse.find(self.filter)
            for i, j, v in zip(I, J, np.sign(V)):
                m[gene[i],gene[j]] = v
            # OPTION: add the first moving gene
            m[0,gene[g1]] = 1
            # OPTION: remove all self-interactions
            m[range(G0),range(G0)] = 0
            # OPTION: keep stimulus everywhere
            # m[0,1:] = 1
            if not dense:
                inter = {t: 0*sparse.csc_matrix(m) for t in times}
                mask = sparse.csc_matrix(m)
            else: mask = m
        else: mask = None
        if self.filter is None or dense:
            inter = {t: np.zeros((G0,G0)) for t in times}

        if verb:
            print('Filtering mask:')
            print(sparse.csc_matrix(mask))

        # Inference procedure
        y, q = inference(x, inter, basal, a, b, c, mask, l=l, tol=tol,
            max_iter=max_iter, sign=sign, verb=verb)

        # Build the results
        self.inter_time = {t: sparse.dok_matrix((G,G)) for t in times}
        self.basal = np.zeros(G)
        for i, g in enumerate(genes):
            self.basal[g] = basal[i]
        self.inter = sparse.dok_matrix((G,G))
        for i, j in zip(*mask.nonzero()):
            gi, gj = genes[i], genes[j]
            s = np.array([np.sign(inter[t][i,j]) for t in times - {0}])
            f = np.array([np.abs(inter[t][i,j]) for t in times - {0}])
            fmax = np.max(f)
            if fmax > 0:
                self.inter[gi,gj] = np.mean(s[f==fmax]) * fmax
            for t in times:
                self.inter_time[t][gi,gj] = inter[t][i,j]
        self.y = y
        self.q = q
        
    def simulate(self, t, burnin=None, verb=False):
        """
        Perform simulation of the network model (bursty PDMP version).
        """
        # Check parameters
        test = ((self.a is None) + (self.d is None)
                + (self.basal is None) + (self.inter is None))
        # Prepare time points
        if test: raise ValueError('Model parameters not yet specified')
        if self.inter is None: print('Interactions must be specified')
        if np.size(t) == 1: t = np.array([t])
        if np.any(t != np.sort(t)):
            raise ValueError('Time points must appear in increasing order')
        # Get the list of gene indices
        if self.genes is None:
            raise ValueError('genes not yet provided')
        v = [0] + self.genes
        # Case 1 (no filtering): all genes are simulated
        if self.filter is None:
            a = self.a
            d = self.d
            basal = self.basal
            inter = self.inter.toarray()
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
        network = BurstyPDMP(a, d, basal, inter)
        # Burnin simulation without stimulus
        if burnin is not None:
            network.simulation([burnin], verb=verb)
        # Activate the stimulus
        network.state['P'][0] = 1
        # Final simulation with stimulus
        sim = network.simulation(t, verb=verb)
        m, p = sim['M'], sim['P']
        return Simulation(self.genes, t, m, p)

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
            inter = self.inter.toarray()
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

    def print_filter(self):
        """
        Print filtered interactions.
        """
        I, J = self.filter.nonzero()
        for i, j in zip(I,J):
            if self.filter[i,j] > 0: print('{} -> {}'.format(i,j))
            elif self.filter[i,j] < 0: print('{} -| {}'.format(i,j))

    def plot_obj(self, file=None):
        from harissa.graphics import plot_obj as plot
        plot(self.q, file=file)

    def plot_xy(self, data, g1=1, g2=2, time=True, file=None):
        from harissa.graphics import plot_xy as plot
        plot(data, self.y, g1=g1, g2=g2, time=time, file=file)


    def preprocess(self, path=''):
        """
        Perform preprocessing and save the results.
        """
        pass

    def load_preprocess(self, path):
        """
        Load preprocessing output files.
        """
        pass


class Cascade(NetworkModel):
    """
    Particular network with a cascade shape.
    Essentially useful for simulation.
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
    Random network with a tree shape.
    Essentially useful for simulation.
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

