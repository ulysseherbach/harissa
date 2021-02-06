"""
harissa.inference
-----------------

Inference of the network model.
"""
from .network import inference
from .kinetics import infer_kinetics

__all__ = ['inference', 'infer_kinetics']
