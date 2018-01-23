"""Classes to handle the stochastic gene network model"""

import numpy as np
from . import core


class GeneNetwork:
    """
    A general class to handle the stochastic network model.

    Two modes:
        - the full description which includes promoter states
        - the bursty limit regime which is faster to compute

    Class attributes
    ----------------
    size : positive int
        Number of genes in the network.

    param : structured array ('S0', 'D0', 'S1', 'D1')
        S0[i] is the creation rate of mRNA i
        D0[i] is the degradation rate of mRNA i
        S1[i] is the creation rate of protein i
        D1[i] is the degradation rate of protein i

    regime : string, 'bursty' (default) or 'general'
        'general' is the general two-state model with promoters
        'bursty' is the regime where promoters are not described

    state : structured array ('E', 'M', 'P') or ('M', 'P')
        state['E'][i] contains state of promoter i (if regime is 'general')
        state['M'][i] contains state of mRNA i
        state['P'][i] contains state of protein i
    """
    def __init__(self, size, param, regime, state):
        self.size = size
        self.param = param
        self.regime = regime
        self.state = state
        ### Set default simulation parameters
        if not hasattr(self,'thin_cst'):
            self.thin_cst = 10*np.sum(self.param['D0'])
        if not hasattr(self,'euler_step'):
            self.euler_step = 0.01/np.max(self.param['D0'])

    def kon(self, P):
        """Interaction function kon (off->on rate),
        given current protein levels P.
        
        NB: This is an arbitrary default constant value."""
        return 1*np.sum(self.param['D0'])

    def koff(self, P):
        """Interaction function kon (on->off rate),
        given current protein levels P.
        
        NB: This is an arbitrary default constant value."""
        return 10*np.sum(self.param['D0'])

    def simulate(self, timepoints, method='exact', info=False):
        """Simulation of the network

        Parameters
        ----------
        timepoints : array, must be sorted in increasing order
            Time-points for which the simulation will be recorded.
        """
        if (np.size(timepoints) == 1):
            timepoints = np.array([timepoints])

        if np.any(timepoints != np.sort(timepoints)):
            print('Error: timepoints must be in increasing order')
            return None
        
        if (method == 'exact'):
            if (self.regime == 'general'):
                return core.sim_exact_full(self, timepoints, info)
            elif (self.regime == 'bursty'):
                return core.sim_exact_bursty(self, timepoints, info)
        elif (method == 'ode'):
            return core.sim_ode(self, timepoints)


class AutoActiv(GeneNetwork):
    """
    A network model with 'Hill-like' interactions.

    NB: This class inherits from the GeneNetwork class.

    Parameter
    ---------
    size : positive int
        Number of genes in the network.

    Inherited attributes (see GeneNetwork)
    --------------------------------------
    size, state, S0, D0, S1, D1

    Specific attributes
    -------------------
    K0 : array of positive floats
        Lower bounds for kon.
    
    K1 : array of positive floats
        Upper bounds for kon.

    B : array of positive floats
        Constant values for koff

    M : square matrix/array of nonnegative floats
        Values for the Hill power coefficients

    S : square matrix/array of positive floats
        Values for the Hill threshold coefficients

    theta : square matrix/array of floats
        theta[i,j] is the strength of interaction j -> i
    """
    def __init__(self, size, regime='bursty'):
        G = size
        ### Define default global parameters
        types = [('S0','float64'), ('D0','float64'),
                 ('S1','float64'), ('D1','float64')]
        param = np.array([(1,1,0.2,0.2) for i in range(G)], dtype=types)
        ### Model-specific basal parameters
        self.K0 = 0.2*param['D0']
        self.K1 = 2*param['D0']
        self.B = 5*param['D0']
        self.M = 2*np.ones((G,G))
        self.S = 0.1*np.ones((G,G))
        ### Interaction matrix
        self.theta = np.zeros((G,G))
        ### State of the system
        if (regime == 'bursty'):
            types = [('M','float64'), ('P','float64')]
            state = np.array([(0,0) for i in range(size)], dtype=types)
            self.thin_cst = np.sum(self.K1) # Only correct if K1 > K0
            self.euler_step = 0.1/np.max(self.K1)
        elif (regime == 'general'):
            types = [('E','uint8'), ('M','float64'), ('P','float64')]
            state = np.array([(0,0,0) for i in range(size)], dtype=types)
            self.thin_cst = np.sum(self.B) # Only correct if B > K1
            self.euler_step = 0.1/np.max(self.B)
        else: print('Error: regime must be either "general" or "bursty".')
        ### Finally set the global GeneNetwork default attributes
        GeneNetwork.__init__(self, size, param, regime, state)

    def kon(self, P):
        """Interaction function kon (off->on rate),
        given current protein levels P.

        NB: Form of the AutoActiv class."""
        G = self.size
        a, s, m = np.exp(self.theta), self.S, self.M
        vP = P*np.ones((G,1))
        x = (vP/s)**m
        I = np.ones((G,G)) - np.diag(np.ones(G))
        Phi = np.prod((I + a*x)/(1 + I*x), axis=1)
        k0, k1 = self.K0, self.K1
        return (k0 + k1*Phi)/(1 + Phi)

    def koff(self, P):
        """Interaction function kon (on->off rate),
        given current protein levels P.

        NB: Form of the AutoActiv class.
        In this model, koff does not depend on P."""
        return self.B


### Utility function to define models
def load(obj):
    """Define a model from any object using its attributes."""
    if hasattr(obj,'model'):
        if (obj.model == 'autoactiv'):
            if not hasattr(obj,'regime'):
                obj.regime = 'general'
            network = AutoActiv(obj.G, obj.regime)
            network.param['S0'] = obj.S0
            network.param['D0'] = obj.D0
            network.param['S1'] = obj.S1
            network.param['D1'] = obj.D1
            network.K0 = obj.K0
            network.K1 = obj.K1
            network.B = obj.B
            network.M = obj.M
            network.S = obj.S
            network.theta = obj.theta
            network.thin_cst = obj.thin_cst
            return network
    else:
        print('Error while loading network.')







