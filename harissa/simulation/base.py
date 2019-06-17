"""
Perform simulations
"""

class Simulation:
    """
    Basic object to store simulations
    """
    def __init__(self, genes, t, m, p):
        self.genes = genes # Gene indices
        self.t = t # Time points
        self.m = m # mRNAs
        self.p = p # Proteins
    
    def plot(self, file=None):
        from ..graphics import plot_sim
        plot_sim(self.t, self.m, self.p, self.genes, file=file)
        