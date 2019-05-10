"""
Harissa
-------

Tools for mechanistic-based gene network inference

Current subpackages
-------------------
autoactiv: auto-activation inference (network hyperparameters)
automodel: network inference using a mechanistic-based random field
grnsim: stochastic simulation of the corresponding dynamical model
hartree: network inference using a self-consistent proteomic field

Author: Ulysse Herbach (ulysse.herbach@inria.fr)
"""
__version__ = '0.2'
__all__ = []

from .hartree import SingleCellData
__all__ += ['SingleCellData']