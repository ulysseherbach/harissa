# Generate data for a basic network
import sys; sys.path += ['../']
import numpy as np
from harissa import NetworkModel

np.random.seed(0)

# Number of cells
C = 10

# Time points
k = np.linspace(0, C, 11, dtype='int')
t = np.linspace(0, 20, 10, dtype='int')
time = np.zeros(C, dtype='int')
for i in range(10):
    time[k[i]:k[i+1]] = t[i]

# Number of genes
G = 4

# Prepare data
data = np.zeros((C+1,G+2), dtype='int')
data[0][1:] = np.arange(G+1) # Genes
data[1:,0] = time # Time points
data[1:,1] = 1 * (time > 0) # Stimulus
    
# Initialize the model
model = NetworkModel(4)
model.d[0] = 1
model.d[1] = 0.2
model.basal[1:] = -5
model.inter[0,1] = 10
model.inter[1,2] = 10
model.inter[1,3] = 10
model.inter[3,4] = 10
model.inter[4,1] = -10
model.inter[2,2] = 10
model.inter[3,3] = 10

# Generate data
for k in range(C):
    # print(f'* Cell {k+1}')
    sim = model.simulate(time[k], burnin=5)
    data[k+1,2:] = np.random.poisson(sim.m[0])

# Save data in somewhat standard format
fname = f'test_data.txt'
np.savetxt(fname, data.T, fmt='%d', delimiter='\t')
