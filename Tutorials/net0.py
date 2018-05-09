"""General template for testing simulations"""
import numpy as np

### Global settings
G = 3 # Number of genes
S0 = 1*np.ones(G) # mRNA creation rates
D0 = 1*np.ones(G) # mRNA degradation rates
S1 = 1*np.ones(G) # Protein creation rates
D1 = 0.2*np.ones(G) # Protein degradation rates

### Specific model parameters
K0 = 0.1*np.ones(G)
K1 = 3.1*np.ones(G)
B = 10*np.ones(G)
M = 3*np.ones((G,G))
S = 0.12*np.outer(np.ones(G), S0*S1/(D0*D1))

### Interactions
inter = {(1,1): 5, (1,2): 5, (2,3): 5}