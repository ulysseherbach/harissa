"""A class to handle the gamma-binomial auto-model"""
import numpy as np
from . import config as cf
from . import utils as ut
from . import core
from .vtheta import Vtheta
from .result import Result

class AutoModel:
    """A class for handling the gamma-binomial auto-model."""
    def __init__(self, theta, a, c, em_count=0):
        self.size = np.size(c)
        self.theta = theta.copy()
        self.hyperparam = (a.copy(),c.copy())
        self.em_count = em_count

    def __repr__(self):
        return 'AutoModel for {} genes'.format(self.size)

    def sample(self, ncell=None, x0=None, z0=None, iter_gibbs=10):
        """Get approximate samples from the model."""
        G = self.size # Number of genes
        a, c = self.hyperparam
        theta = self.theta
        ### Check for a number of cells
        if ncell is None: ncell = 1
        ### Check for x0 and z0
        if (x0 is None) or (z0 is None):
            x0 = np.zeros((ncell,G))
            z0 = np.zeros((ncell,G), dtype=('uint8'))
        ### Check the sizes of x0 and z0
        if ((np.size(x0[0,:]) != G) or (np.size(z0[0,:]) != G)):
            print('Error: sizes of x0 and z0 must match the model.')
        elif (np.size(x0[:,0]) != np.size(z0[:,0])):
            print('Error: sizes of x0 and z0 must match together.')
        else:
            x = x0.copy()
            z = z0.copy()
            core.gibbs_sample(x, z, iter_gibbs, a, theta, c)
            if (ncell == 1): x, z = x[0], z[0]
            return (x,z)

    def infer(self, data, nsteps=1, method='sp', traj_theta=False,
            sample=None, fname=None, name=None, ncache=None, info=True):
        """Infer theta from data given hyperparameters a and c.
        This modifies the theta attribute of the input.
        
        Parameters
        ----------
        data : array of floats
            Each row is a cell, each column is a gene.
            Missing values can be encoded by "-1".
        """
        C, G = np.shape(data)
        a, c = self.hyperparam
        theta = self.theta
        alpha = np.zeros((C,G)) # Variational parameters
        vq = np.zeros(nsteps+1) # Path of the objective function
        if method == 'mc':
            if sample is None:
                iter_init, K = cf.iter_gibbs_init, cf.sample_size
                x = np.zeros((K,G)) # Samples of X (mRNA)
                z = np.zeros((K,G), dtype=('uint8')) # Samples of Z
            else:
                x, z = sample
                iter_init, K = 0, np.size(z[:,0])
        ### Initialization
        if info: print('Initialization...\n')
        if method == 'mc':
            core.gibbs_sample(x, z, iter_init, a, theta, c)
            # vq[0] = core.log_likelihood_estim(data, z, a)
        elif method in {'bf', 'bf-qn'}:
            states = ut.state_vector(c)
            # vq[0] = ut.log_likelihood_exact(data, a, theta, c)
        elif method == 'sp': message = {}
        if traj_theta or (fname is not None):
            vtheta = Vtheta(theta)
            if fname is not None: vtheta.savetxt(fname)
        else: vtheta = None
        ### Variational EM loop
        for t in range(nsteps):
            msg = 'EM step {}'.format(t+1)
            if method == 'sp':
                ne = int(len(message)/2)
                msg += ' ({} edge{})'.format(ne, (ne>1)*'s')
            if info: print(msg + '...')
            self.em_count += 1
            ### Update alpha given theta and data
            alpha = core.e_step_var(alpha, data, a, theta, c, info)
            ### Update theta given alpha (x and z are modified)
            if method == 'mc':
                theta = core.m_step_mc(x, z, alpha, data, a, theta, c)
                # vq[t+1] = core.log_likelihood_estim(data, z, a)
            elif method == 'bf':
                theta = core.m_step_brute(states,alpha,data,a,theta,c,info)
                # vq[t+1] = ut.log_likelihood_exact(data, a, theta, c)
            elif method == 'bf-qn':
                theta = core.m_step_bf_qn(states,alpha,data,a,theta,c,info)
            elif method == 'sp':
                theta = core.m_step_sp(message,alpha,data,a,theta,c,info)
            ### Optionally store theta values
            if traj_theta or (fname is not None):
                vtheta.append(theta)
                if fname is not None: vtheta.savetxt(fname)
            ### Show estimated entropy
            # print('Entropy: {}'.format(entropy_mc(z)))
            if ncache is None: ctest = False
            else: ctest = ((t+1)%ncache == 0)
            endtest = (t == nsteps-1)
            if (ctest or endtest):
                ### Update the model and return the results
                self.theta = theta
                kwargs = {'traj_logl': vq, 'traj_theta': vtheta}
                if method == 'mc':
                    sample = (x,z)
                    # kwargs['ent_mc'] = ut.entropy_mc(z)
                    # kwargs['ent_varmc'] = ut.entropy_varmc(z, alpha, c)
                elif method == 'bf' or method == 'bf-qn':
                    sample = None
                    # kwargs['ent_varbrute'] = ut.entropy_varbrute(alpha, c)
                kwargs['em_count'] = self.em_count
                res = Result(theta, alpha, sample, method, **kwargs)
                # if path and name: res.save(path, name)
        return res

### Utility functions for automodel objects
def save(file, model):
    """Save an automodel network into a text file"""
    G = model.size
    a, c = model.hyperparam
    net = np.zeros((G,G+4))
    net[:,0] = c
    net[:,1] = a[0,:]
    net[:,2] = a[1,:]
    net[:,3] = a[2,:]
    net[:,4:] = model.theta
    flist = ['%u'] + 3*['%.5f'] + + G*['%.2f']
    np.savetxt(file, net, fmt=flist)

def load(file):
    """Load an automodel object from a formatted text file"""
    net = np.loadtxt(file)
    c = net[:,0]
    G = np.size(c)
    a = np.zeros((3,G))
    a[0,:] = net[:,1]
    a[1,:] = net[:,2]
    a[2,:] = net[:,3]
    theta = net[:,4:]
    check = True
    if (np.shape(theta) != (G,G)): check = False
    if not ut.is_valid_theta(theta): check = False
    if check: return AutoModel(theta, a, c)
    else: print('Error: automodel file not well formated.')