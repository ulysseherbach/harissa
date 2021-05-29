### Using the autoactiv package ###

import sys; sys.path.append("../")
import numpy as np 
import harissa.grnsim as ns
import harissa.autoactiv as av


### 1. The quick approach

### Generate fake data
N = 10 # Number of cells to simulate
data = np.zeros((N,2))
data[:,1] = np.random.gamma(2, size=N)

### Infer parameters of the autoactiv model
model = av.infer(data)
print(model.a)
print(model.c)


### 2. More sophisticated approach

### Generate some example data
N = 10 # Number of cells to simulate
timepoints = (0, 8, 24, 33, 48, 72)
import net0
network = ns.networks.load(net0, mode='bursty')
network.state['M'] = 1
### Structure: (idcell, timepoint, gene 1, gene 2...)
types = [('idcell', 'int64'), ('timepoint', 'float64'),
    ('Gene 1', 'float64'), ('Gene 2', 'float64')]
lcells = []
for k in range(N):
    simu = network.simulate(timepoints)
    M = simu['M']
    idcell = k + 1
    for i, t in enumerate(timepoints):
        lcells += [(idcell, t, M[i,0], M[i,1])]
cells = np.array(lcells, dtype=types)
### Create a scdata object for the autoactiv package
data = av.scdata(cells)
print(data)

### Inference
p = av.posterior(data)

### Option 1: get the full results as a dictionary of models
modeldict = av.mapestim(data, p)
print(modeldict['Gene 1'], modeldict['Gene 2'])

### Option 2: directly get hyperparameters a and c
a, dtheta, c = av.get_param(data, p, timepoints)
print(a, c)