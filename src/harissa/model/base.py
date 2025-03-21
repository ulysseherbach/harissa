"""Main class for network inference and simulation."""
import numpy as np
from harissa.model.cascade import cascade
from harissa.model.tree import tree
from harissa.inference import infer_kinetics, infer_proteins


########################
# Main interface class #
########################


class NetworkModel:
    """Handle networks within Harissa."""

    def __init__(self, n_genes=None):
        # Kinetic parameters
        self.a = None
        self.d = None
        # Network parameters
        self.basal = None
        self.inter = None
        # Default behaviour
        if n_genes is not None:
            n = n_genes + 1  # Genes plus stimulus
            # Default bursting parameters
            self.a = np.zeros((3, n))
            self.a[0] = 0  # Minimal Kon rate (normalized)
            self.a[1] = 2  # Maximal Kon rate (normalized)
            self.a[2] = 0.02  # Inverse burst size of mRNA
            # Default degradation rates
            self.d = np.zeros((2, n))
            self.d[0] = np.log(2)/9  # mRNA degradation rates (per hour)
            self.d[1] = np.log(2)/46  # protein degradation rates (per hour)
            # Default network parameters
            self.basal = np.zeros(n)
            self.inter = np.zeros((n, n))

    def get_kinetics(self, data, verb=False):
        """Compute the basal parameters of all genes."""
        times = data[:, 0]
        n = data[0].size
        # Kinetic values for each gene
        a = np.ones((3, n))
        for g in range(1, n):
            if verb:
                print(f'Calibrating gene {g}...')
            x = data[:, g]
            at, b = infer_kinetics(x, times, verb=verb)
            a[0, g] = np.min(at)
            a[1, g] = np.max(at)
            a[2, g] = b
        self.a = a

    def fit(self, data, l=1, tol=1e-5, verb=False, use_numba=True):
        """Fit the network model to the data."""
        x = data
        # Time points
        times = np.sort(list(set(x[:, 0])))
        self.times = times
        # Default degradation rates
        n = x.shape[1]
        d = np.zeros((2, n))
        d[0] = np.log(2)/9  # mRNA degradation rates
        d[1] = np.log(2)/46  # protein degradation rates
        self.d = d
        # Kinetic parameters
        self.get_kinetics(data, verb)
        a = self.a
        # Concentration parameter
        c = 100 * np.ones(n)
        # Get protein levels
        y = infer_proteins(x, a)
        self.y = y
        # Import necessary modules
        if use_numba:
            from harissa.inference.network_fast import infer_network
        else:
            from harissa.inference.network import infer_network
        # Inference procedure
        theta = infer_network(x, y, a, c, l, tol, verb)
        # Build the results
        self.basal = np.zeros(n)
        self.inter = np.zeros((n, n))
        self.basal_time = {time: np.zeros(n) for time in times}
        self.inter_time = {time: np.zeros((n, n)) for time in times}
        for t, time in enumerate(times):
            self.basal_time[time][:] = theta[t][:, 0]
            self.inter_time[time][:, 1:] = theta[t][:, 1:]
        self.basal[:] = theta[-1][:, 0]
        self.inter[:, 1:] = theta[-1][:, 1:]

    def simulate(self, t, burnin=None, use_numba=False, **kwargs):
        """
        Perform simulation of the network model (bursty PDMP version).
        """  # noqa: D200
        # Get keyword arguments
        m0 = kwargs.get('M0')
        p0 = kwargs.get('P0')
        verb = kwargs.get('verb', False)
        # Check parameters
        check = ((self.a is None) + (self.d is None)
                + (self.basal is None) + (self.inter is None))
        # Prepare time points
        if check:
            msg = 'Model parameters not yet specified'
            raise ValueError(msg)
        if np.size(t) == 1:
            t = np.array([t])
        if np.any(t != np.sort(t)):
            msg = 'Time points must appear in increasing order'
            raise ValueError(msg)
        # Import necessary modules
        from harissa.simulation.base import Simulation
        if use_numba:
            from harissa.simulation.pdmp_fast import BurstyPDMP
        else:
            from harissa.simulation.pdmp import BurstyPDMP
        a = self.a
        d = self.d
        basal = self.basal
        inter = self.inter
        network = BurstyPDMP(a, d, basal, inter)
        # Burnin simulation without stimulus
        if m0 is not None:
            network.state['M'][1:] = m0[1:]
        if p0 is not None:
            network.state['P'][1:] = p0[1:]
        if burnin is not None:
            network.simulation([burnin], verb)
        # Activate the stimulus
        network.state['P'][0] = 1
        # Final simulation with stimulus
        sim = network.simulation(t, verb)
        m, p = sim['M'], sim['P']
        return Simulation(t, m, p)

    def simulate_ode(self, t, burnin=None, **kwargs):
        """
        Perform simulation of the network model (ODE version).

        This is the slow-fast limit of the PDMP model, which is only
        relevant when promoters & mRNA are much faster than proteins.
        p: solution of a nonlinear ODE system involving proteins only
        m: mean mRNA levels given protein levels (quasi-steady state)
        """
        # Get keyword arguments
        m0 = kwargs.get('M0')
        p0 = kwargs.get('P0')
        verb = kwargs.get('verb', False)
        # Check parameters
        check = ((self.a is None) + (self.d is None)
                + (self.basal is None) + (self.inter is None))
        # Prepare time points
        if check:
            msg = 'Model parameters not specified yet'
            raise ValueError(msg)
        if self.inter is None:
            print('Interactions must be specified')
        if np.size(t) == 1:
            t = np.array([t])
        if np.any(t != np.sort(t)):
            msg = 'Time points must appear in increasing order'
            raise ValueError(msg)
        # Import necessary modules
        from harissa.simulation.base import Simulation
        from harissa.simulation.ode import ApproxODE
        a = self.a
        d = self.d
        basal = self.basal
        inter = self.inter
        network = ApproxODE(a, d, basal, inter)
        # Burnin simulation without stimulus
        if m0 is not None:
            network.state['M'][1:] = m0[1:]
        if p0 is not None:
            network.state['P'][1:] = p0[1:]
        if burnin is not None:
            network.simulation([burnin], verb)
        # Activate the stimulus
        network.state['P'][0] = 1
        # Final simulation with stimulus
        sim = network.simulation(t, verb)
        m, p = sim['M'], sim['P']
        return Simulation(t, m, p)


###########################
# Special network classes #
###########################


class Cascade(NetworkModel):
    """Particular network with a cascade structure."""

    def __init__(self, n_genes, autoactiv=False):
        # Get NetworkModel default features
        NetworkModel.__init__(self, n_genes)
        # New network parameters
        basal, inter = cascade(n_genes)
        if autoactiv:
            for i in range(1, n_genes + 1):
                inter[i, i] = 5
        self.basal = basal
        self.inter = inter


class Tree(NetworkModel):
    """Random network with a tree structure."""

    def __init__(self, n_genes, autoactiv=False):
        # Get NetworkModel default features
        NetworkModel.__init__(self, n_genes)
        # New network parameters
        basal, inter = tree(n_genes)
        if autoactiv:
            for i in range(1, n_genes + 1):
                inter[i, i] = 5
        self.basal = basal
        self.inter = inter
