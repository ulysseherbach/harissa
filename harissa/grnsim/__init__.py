"""
Simulation of Gene Regulatory Networks

Networks are modelled by piecewise-deterministic Markov processes (PDMP)

Two simulation options:
- exact (thinning method)

TODO:
- put the thinning constant and the euler step outside the model class
- implement a check on these constants before simulation
"""
__version__ = '0.2'
__author__ = 'Ulysse Herbach'
__all__ = []

from .model import load
from .graphics import plotsim, histo
__all__ += ['load', 'plotsim', 'histo']