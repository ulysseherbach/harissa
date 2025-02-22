"""
Generate cascade networks
"""
import numpy as np

def cascade(n_genes):
    """
    Generate a simple activation cascade (1) -> (2) -> ... -> (n_genes).
    """
    G = n_genes + 1
    basal = np.zeros(G)
    inter = np.zeros((G,G))
    basal[1:] = -5 # Low basal level of downstream genes
    for i in range(n_genes):
        inter[i,i+1] = 10
    return basal, inter


# Tests
if __name__ == '__main__':
    basal, inter = cascade(5)
    print(inter)
