"""
harissa.inference
-----------------

Inference of the network model.
"""
from .kinetics import infer_kinetics
from .network import infer_proteins, infer_network

__all__ = ['infer_kinetics', 'infer_proteins', 'infer_network']
