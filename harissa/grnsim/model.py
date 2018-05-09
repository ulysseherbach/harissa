"""Classes to handle the stochastic gene network model"""
import numpy as np
from . import core
from .utils import theta_base

### General network form
class GeneNetworkFull:
    """
    A general class to handle the stochastic network model.
    NB: this is the full description including promoter states.

    Input
    -----
    size : positive int
        Number of genes in the network.

    param : structured array ('S0', 'D0', 'S1', 'D1') with length [size]
        param['S0'][i] is the creation rate of mRNA i
        param['D0'][i] is the degradation rate of mRNA i
        param['S1'][i] is the creation rate of protein i
        param['D1'][i] is the degradation rate of protein i

    state : structured array ('E', 'M', 'P')
        state['E'][i] contains state of promoter i
        state['M'][i] contains state of mRNA i
        state['P'][i] contains state of protein i

    thin_cst : thinning constant for exact stochastic simulations
    euler_step : euler step size for the deterministic version

    Class attributes
    ----------------
    param, state : as above
    kon, koff : interaction functions of the model
    simulate : simulation function
    """
    def __init__(self, size, param, state, thin_cst, euler_step):
        self.param = param
        self.state = state
        self.thin_cst = thin_cst
        self.euler_step = euler_step

    def kon(self, P):
        """Interaction function kon (off->on rate), given protein levels P.
        NB: This is an arbitrary default constant value."""
        return 1*np.sum(self.param['D0'])

    def koff(self, P):
        """Interaction function kon (on->off rate), given protein levels P.
        NB: This is an arbitrary default constant value."""
        return 10*np.sum(self.param['D0'])

    def simulate(self, timepoints, method='exact', info=False):
        """Simulation of the network

        Parameters
        ----------
        timepoints : array, must be sorted in increasing order
            Time-points for which the simulation will be recorded.
        """
        if np.size(timepoints) == 1: timepoints = np.array([timepoints])
        if np.any(timepoints != np.sort(timepoints)):
            raise ValueError('timepoints must be in increasing order')
        if (method == 'exact'):
            return core.sim_exact_full(self, timepoints, info)
        elif (method == 'ode'):
            return core.sim_ode(self, timepoints)
        else: raise ValueError('method must be either "exact" or "ode"')


class GeneNetworkBursty:
    """
    A general class to handle the stochastic network model.
    NB: this is the bursty limit regime (no promoter, faster to compute)
    
    Input
    -----
    size : positive int
        Number of genes in the network.

    param : structured array ('S0', 'D0', 'S1', 'D1') with length [size]
        param['S0'][i] is the creation rate of mRNA i
        param['D0'][i] is the degradation rate of mRNA i
        param['S1'][i] is the creation rate of protein i
        param['D1'][i] is the degradation rate of protein i

    state : structured array (M', 'P')
        state['M'][i] contains state of mRNA i
        state['P'][i] contains state of protein i

    thin_cst : thinning constant for exact stochastic simulations
    euler_step : euler step size for the deterministic version

    Class attributes
    ----------------
    param, state : as above
    kon, koff : interaction functions of the model
    simulate : simulation function
    """
    def __init__(self, size, param, state, thin_cst, euler_step):
        self.param = param
        self.state = state
        self.thin_cst = thin_cst
        self.euler_step = euler_step

    def kon(self, P):
        """Interaction function kon (off->on rate), given protein levels P.
        NB: This is an arbitrary default constant value."""
        return 1*np.sum(self.param['D0'])

    def koff(self, P):
        """Interaction function kon (on->off rate), given protein levels P.
        NB: This is an arbitrary default constant value."""
        return 10*np.sum(self.param['D0'])

    def simulate(self, timepoints, method='exact', info=False):
        """Simulation of the network

        Parameters
        ----------
        timepoints : array, must be sorted in increasing order
            Time-points for which the simulation will be recorded.
        """
        if np.size(timepoints) == 1: timepoints = np.array([timepoints])
        if np.any(timepoints != np.sort(timepoints)):
            raise ValueError('timepoints must be in increasing order')
        if (method == 'exact'):
            return core.sim_exact_bursty(self, timepoints, info)
        elif (method == 'ode'):
            return core.sim_ode(self, timepoints)


### A possible network model with a particular interaction form
class AutoActivFull(GeneNetworkFull):
    """
    A network model with 'Hill-like' interactions.
    NB: this is the full description including promoter states.

    Parameter
    ---------
    size : positive int
        Number of genes in the network.

    Inherited attributes (see GeneNetworkFull)
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

    inter : square matrix/array of floats
        - if i != j, inter[i,j] is the strength of interaction i -> j
        - if i == j this is rather a modification of basal promoter activity.
    """
    def __init__(self, size):
        ### Define default global parameters
        G = size
        S0, D0 = 1*np.ones(G), 1*np.ones(G)
        S1, D1 = 0.2*np.ones(G), 0.2*np.ones(G)
        ### Model-specific basal parameters
        K0, K1, B = 0.2*D0, 2.2*D0, 5*D0
        M, S = 2*np.ones((G,G)), 0.1*np.ones((G,G))
        self.K0, self.K1, self.B, self.M, self.S = K0, K1, B, M, S
        ### Interactions
        self.inter = {}
        ### State of the system
        types = [('E','uint'), ('M','float'), ('P','float')]
        state = np.array([(0,0,0) for i in range(G)], dtype=types)
        ### Simulation parameters
        thin = np.sum(self.B) # Only correct if B > K1
        euler = 0.1/np.max(self.B)
        ### Finally set the global GeneNetworkFull default attributes
        types = [('S0','float'), ('D0','float'),
                 ('S1','float'), ('D1','float')]
        param = np.array(G*[(S0[0], D0[0], S1[0], D1[0])], dtype=types)
        GeneNetworkFull.__init__(self, size, param, state, thin, euler)

    def get_theta_base(self):
        """Get the basal theta of the model. This is the diagonal value
        for which genes are independent and somewhat well balanced. """
        S0, D0 = self.param['S0'], self.param['D0']
        S1, D1 = self.param['S1'], self.param['D1']
        K0, K1, B, M, S = self.K0, self.K1, self.B, self.M, self.S
        return theta_base(S0, D0, S1, D1, K0, K1, B, M, S)

    def get_theta(self):
        """Get the effective theta of the model."""
        theta0 = self.get_theta_base()
        for (i,j) in self.inter.keys():
            theta0[j-1,i-1] += self.inter[i,j]
        return theta0

    def kon(self, P):
        """Interaction function kon (off->on rate), given protein levels P.
        NB: This is the specific form of the AutoActiv class."""
        theta = self.get_theta()
        G = np.size(theta[0])
        a, s, m = np.exp(theta), self.S, self.M
        vP = P*np.ones((G,1))
        x = (vP/s)**m
        I = np.ones((G,G)) - np.diag(np.ones(G))
        Phi = np.prod((I + a*x)/(1 + I*x), axis=1)
        k0, k1 = self.K0, self.K1
        return (k0 + k1*Phi)/(1 + Phi)

    def koff(self, P):
        """Interaction function kon (on->off rate), given protein levels P.
        NB: This is the specific form of the AutoActiv class.
        In this model, koff actually does not depend on P."""
        return self.B


class AutoActivBursty(GeneNetworkBursty):
    """
    A network model with 'Hill-like' interactions.
    NB: this is the bursty limit regime (no promoter, faster to compute)

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

    inter : square matrix/array of floats
        - if i != j, inter[i,j] is the strength of interaction i -> j
        - if i == j this is rather a modification of basal promoter activity.
    """
    def __init__(self, size):
        ### Define default global parameters
        G = size
        S0, D0 = 1*np.ones(G), 1*np.ones(G)
        S1, D1 = 0.2*np.ones(G), 0.2*np.ones(G)
        ### Model-specific basal parameters
        K0, K1, B = 0.2*D0, 2.2*D0, 5*D0
        M, S = 2*np.ones((G,G)), 0.1*np.ones((G,G))
        self.K0, self.K1, self.B, self.M, self.S = K0, K1, B, M, S
        ### Interactions
        self.inter = {}
        ### State of the system
        types = [('M','float'), ('P','float')]
        state = np.array([(0,0) for i in range(G)], dtype=types)
        ### Simulation parameters
        thin = np.sum(K1) # Only correct if K1 > K0
        euler = 0.1/np.max(self.B)
        ### Finally set the global GeneNetworkBursty default attributes
        types = [('S0','float'), ('D0','float'),
                 ('S1','float'), ('D1','float')]
        param = np.array(G*[(S0[0], D0[0], S1[0], D1[0])], dtype=types)
        GeneNetworkBursty.__init__(self, size, param, state, thin, euler)

    def get_theta_base(self):
        """Get the basal theta of the model. This is the diagonal value
        for which genes are independent and somewhat well balanced. """
        S0, D0 = self.param['S0'], self.param['D0']
        S1, D1 = self.param['S1'], self.param['D1']
        K0, K1, B, M, S = self.K0, self.K1, self.B, self.M, self.S
        return theta_base(S0, D0, S1, D1, K0, K1, B, M, S)

    def get_theta(self):
        """Get the effective theta of the model."""
        theta0 = self.get_theta_base()
        for (i,j) in self.inter.keys():
            theta0[j-1,i-1] += self.inter[i,j]
        return theta0

    def kon(self, P):
        """Interaction function kon (off->on rate), given protein levels P.
        NB: This is the specific form of the AutoActiv class."""
        theta = self.get_theta()
        G = np.size(theta[0])
        a, s, m = np.exp(theta), self.S, self.M
        vP = P*np.ones((G,1))
        x = (vP/s)**m
        I = np.ones((G,G)) - np.diag(np.ones(G))
        Phi = np.prod((I + a*x)/(1 + I*x), axis=1)
        k0, k1 = self.K0, self.K1
        return (k0 + k1*Phi)/(1 + Phi)

    def koff(self, P):
        """Interaction function kon (on->off rate), given protein levels P.
        NB: This is the specific form of the AutoActiv class.
        In this model, koff actually does not depend on P."""
        return self.B


### Load an auto-activation based model
def load(obj, mode='full'):
    """
    Define a network model from any object using its attributes.
    
    Input
    -----
    obj : any object with the right attributes
    mode : 'full' (default) or 'bursty'
    """
    if mode == 'full': network = AutoActivFull(obj.G)
    elif mode == 'bursty': network = AutoActivBursty(obj.G)
    else: raise ValueError('mode must be either "full" or "bursty"')
    network.param['S0'] = obj.S0
    network.param['D0'] = obj.D0
    network.param['S1'] = obj.S1
    network.param['D1'] = obj.D1
    network.K0 = obj.K0
    network.K1 = obj.K1
    network.B = obj.B
    network.M = obj.M
    network.S = obj.S
    network.inter = obj.inter
    if mode == 'full': network.thin_cst = np.sum(obj.B)
    elif mode == 'bursty': network.thin_cst = np.sum(obj.K1)
    return network