# Perform simple data binarization using the `infer_proteins` function
import numpy as np
import sys; sys.path += ['../']
from harissa import NetworkModel
from harissa.inference import infer_proteins

# Import raw data (run network4.py)
data = np.loadtxt('network4_data.txt', dtype=int, delimiter='\t')
C, G = data.shape

# Store binarized data
newdata = np.zeros((C,G), dtype='int')
newdata[:,0] = data[:,0] # Time points

# Calibrate the mechanistic model
model = NetworkModel()
model.get_kinetics(data)

# Get binarized values (gene-specific thresholds)
newdata[:,1:] = infer_proteins(data, model.a)[:,1:].astype(int)

# Save binarized data
np.savetxt(f'test_binarize.txt', newdata, fmt='%d', delimiter='\t')

# Note: a wrapper function is available
from harissa.utils import binarize
bdata = binarize(data)
print(f'newdata = binarize(data)? {np.array_equal(newdata, bdata)}')
