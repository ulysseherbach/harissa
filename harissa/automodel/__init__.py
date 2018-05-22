"""
Automodel for single-cell data
---------

Fitting a biologically justified Markov random field.
It can be seen as a "gamma-binomial" auto-model.
"""
__version__ = '0.3'
__author__ = 'Ulysse Herbach'
__all__ = []

import sys
import multiprocessing
import numpy as np
import scipy.special as sp

### Export various functions
from .model import AutoModel, save, load
from .vtheta import load_vtheta
__all__ += ['save', 'load', 'load_vtheta']

from .graphics import plot_em, plot_marginals
__all__ += ['plot_em', 'plot_marginals']

from .utils import neutral_theta, is_symmetric, map_theta
__all__ += ['neutral_theta', 'is_symmetric', 'map_theta']

from . import config
__all__ += ['config']

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)


### Wrappers
__all__ += ['automodel', 'infer']

def automodel(theta, a=None, c=None):
    """Define a gamma-binomial automodel."""
    if not is_symmetric(theta):
        raise ValueError('theta must be a symmetric matrix.')
    n = np.size(theta[0])
    if c is None: c = np.ones(n, dtype='int')
    if a is None:
        a = np.zeros((3,n))
        a[0] = 0.1*np.ones(n)
        a[1] = 3*np.ones(n)/c
        a[2] = (sp.gamma(a[0] + a[1]*c)/sp.gamma(a[0]))**(1/(a[1]*c))
    return AutoModel(theta, a, c)

def infer(data, theta0, a, c, **kwargs):
    """Main inference function for the gamma-binomial automodel."""
    time = kwargs.get('time', None) # Maximum running time
    if np.sum(data == 0) > 0:
        raise ValueError('there should be no zero in the data.')
    theta = theta0.copy()
    model = automodel(theta, a=a, c=c)
    if time is None:
        res = model.infer(data, **kwargs)
        return res
    else:
        kwargs.pop('time')
        def process():
            sys.stdout = Unbuffered(sys.stdout)
            model.infer(data, **kwargs)
        p = multiprocessing.Process(target=process)
        p.start()
        # Usage: join([timeout in seconds])
        p.join(int(time*3600))
        # If thread is active
        if p.is_alive():
            print('\n...Stopping inference! ({}h)'.format(time))
            p.terminate()
            p.join()

### Useful function to add noise to theta
def add_noise(theta, scale=0.1):
    """Add gaussian noise to non-diagonal entries of theta."""
    G = np.size(theta[0])
    for i in range(G):
        for j in range(i+1,G):
            theta[i,j] += np.random.normal(scale=scale)
            theta[j,i] = theta[i,j]
__all__ += ['add_noise']