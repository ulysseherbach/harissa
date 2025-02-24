"""
harissa.inference
-----------------

Inference of the network model.
"""
from harissa.inference.kinetics import infer_kinetics
from harissa.inference.network import infer_proteins

__all__ = ["infer_kinetics", "infer_proteins"]
