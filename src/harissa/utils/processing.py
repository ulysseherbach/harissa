"""
Data processing
"""
import numpy as np
from ..model import NetworkModel
from ..inference import infer_proteins

def binarize(data):
    """
    Return a binarized version of the data using gene-specific thresholds
    derived from the data-calibrated mechanistic model.
    """
    new_data = np.zeros(data.shape, dtype=data.dtype)
    new_data[:,0] = data[:,0] # Time points
    # Calibrate the mechanistic model
    model = NetworkModel()
    model.get_kinetics(data)
    # Get binarized values (gene-specific thresholds)
    new_data[:,1:] = infer_proteins(data, model.a)[:,1:].astype(int)
    return new_data
