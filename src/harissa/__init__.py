"""
Harissa
=======

Tools for mechanistic gene network inference from single-cell data
------------------------------------------------------------------

Mechanistic model-based gene network inference using
a self-consistent proteomic field (SCPF) approximation.
It is analogous to the unrestricted Hartree approximation
in quantum mechanics, applied to gene expression modeled
as a piecewise-deterministic Markov process (PDMP).

The package also includes a simulation module to generate
single-cell data with transcriptional bursting.

Author: Ulysse Herbach (ulysse.herbach@inria.fr)
"""
from importlib.metadata import version as _version
from harissa.model import NetworkModel, Cascade, Tree

__all__ = ["NetworkModel", "Cascade", "Tree"]

try:
    __version__ = _version("harissa")
except Exception:
    __version__ = "unknown version"
