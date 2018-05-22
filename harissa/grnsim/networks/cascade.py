"""General template for testing simulations"""
from .loader import hillnet

def cascade(size, mode='bursty', autoactiv=True):
    """Build a simple activation cascade network 1 -> 2 -> ... -> size."""
    network = hillnet(size, mode=mode, autoactiv=autoactiv)
    network.basal[1] = 5 # External input on the first gene
    for i in range(1,size):
        network.basal[i+1] = -1 # Low basal level of downstream genes
        network.inter[i,i+1] = 5 # Activation cascade
    return network