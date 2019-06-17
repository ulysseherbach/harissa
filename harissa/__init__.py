"""
Harissa
=======

Gene regulatory network inference from single-cell data
-------------------------------------------------------

Mechanistic-based gene network inference using a
self-consistent proteomic field (SCPF) approximation.
It is analogous to the unrestricted Hartree approximation
in quantum mechanics, applied to gene expression modeled
as a piecewise-deterministic Markov process (PDMP).

Author: Ulysse Herbach (ulysse.herbach@inria.fr)
"""
__version__ = '1.0'

from .model import NetworkModel, Cascade, Tree

__all__ = ['NetworkModel', 'Cascade', 'Tree']
