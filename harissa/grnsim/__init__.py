"""
Simulation of Gene Regulatory Networks

Networks are modelled by piecewise-deterministic Markov processes (PDMP)
"""
__version__ = '0.3'
__author__ = 'Ulysse Herbach'
__all__ = []

from .model import load
from .graphics import plotsim, histo
__all__ += ['load', 'plotsim', 'histo']