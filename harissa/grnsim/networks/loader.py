import numpy as np
from ..model import AutoActivFull, AutoActivBursty

def hillnet(size, mode='bursty', autoactiv=True):
    """
    Load an empty network model (default parameters and no interactions),
    with the Hill-based interaction function of [Herbach et al., 2017].
    By default all genes have auto-activation but this is fully configurable.
    """
    if mode == 'full': network = AutoActivFull(size)
    elif mode == 'bursty': network = AutoActivBursty(size)
    if mode in {'full', 'bursty'}:
        if not autoactiv:
            for i in range(size): network.M[i,i] = 0
        return network
    else: raise ValueError('mode must be either "full" or "bursty"')

def load(obj, mode='full'):
    """
    Load a network model from any object using its attributes.
    
    Input
    -----
    obj : any object with the right attributes
    mode : 'full' or 'bursty'
    """
    if mode == 'full':
        network = AutoActivFull(obj.G)
    elif mode == 'bursty':
        network = AutoActivBursty(obj.G)
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
    if hasattr(obj, 'basal'): network.basal = obj.basal
    if hasattr(obj, 'inter'): network.inter = obj.inter
    if mode == 'full':
        network.thin_cst = np.sum(obj.B)
    elif mode == 'bursty':
        network.thin_cst = np.sum(obj.K1)
    return network