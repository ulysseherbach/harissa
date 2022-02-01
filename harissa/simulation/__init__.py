"""
harissa.simulation
------------------

Simulation of the network model.
"""
from .base import Simulation
from .ode import ApproxODE

# Handle Numba as an option
from ..utils import NUMBA_SIM
if NUMBA_SIM: from .pdmp_fast import BurstyPDMP
else: from .pdmp import BurstyPDMP

__all__ = ['Simulation', 'BurstyPDMP', 'ApproxODE']
