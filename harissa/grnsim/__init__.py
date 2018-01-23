"""
Simulation of Gene Regulatory Networks

Networks are modelled by piecewise-deterministic Markov processes (PDMP)

Two simulation options:
- exact (thinning method)
- approximate (hybrid Euler scheme)

TODO:
- put the thinning constant and the euler step outside the model class
- implement a check on these constants before simulation
"""

__all__ = ['load', 'plotsim', 'histo']
__version__ = '0.1'
__author__ = 'Ulysse Herbach'

### Export the model class
from grnsim.model import load

### Export the plot function
from grnsim.graphics import plotsim, histo

### Optionally set the seed for reproducible output
# import numpy as np
# np.random.seed(1)