"""
Simulation of Gene Regulatory Networks

Networks are modelled by piecewise-deterministic Markov processes (PDMP)
"""
__version__ = '0.3'
__author__ = 'Ulysse Herbach'
__all__ = []

from .graphics import plotsim, histo
__all__ += ['plotsim', 'histo']

### Module for builing network examples
from . import networks
__all__ += ['networks']