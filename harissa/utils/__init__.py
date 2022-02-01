"""
harissa.utils
-------------

Various utility functions for the package.
"""

# Use Numba for acceleration
NUMBA_INF = True
NUMBA_SIM = False

# Handle Numba as an option
def identity(func): return func
if NUMBA_INF or NUMBA_SIM: from numba import njit
else: njit = identity

__all__ = ['njit']
