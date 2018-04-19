"""General template for testing simulations"""

import numpy as np

### Global settings
model = 'autoactiv' # Interaction form
regime = 'general' # General regime
G = 3 # Number of genes

### Kinetic parameters
S0 = 1*np.ones(G) # mRNA creation rates
D0 = 1*np.ones(G) # mRNA degradation rates
S1 = 1*np.ones(G) # Protein creation rates
D1 = 0.2*np.ones(G) # Protein degradation rates

### Specific 'autoactiv' parameters
K0 = 0.1*np.ones(G)
K1 = 3.1*np.ones(G)
B = 10*np.ones(G)
M = 3*np.ones((G,G))
S = 0.12*np.ones((G,G))
### Important: normalization of the thresholds
scale = S0*S1/(D0*D1)
for i in range(G):
    S[:,i] = S[:,i]*scale[i]

### Constant for the simulation by thinning
thin_cst = np.sum(B)

### Interaction matrix
### NB: theta[i,j] stands for interaction j -> i
theta = np.zeros((G,G))