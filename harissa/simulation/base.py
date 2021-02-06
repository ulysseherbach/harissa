"""
Perform simulations
"""
import numpy as np

class Simulation:
    """
    Basic object to store simulations
    """
    def __init__(self, t, m, p):
        self.t = t # Time points
        self.m = m # mRNAs
        self.p = p # Proteins
    
    def plot(self, genes=None, file=None):
        from ..graphics import plot_sim
        G = self.m[0].size + 1
        if genes is None: genes = list(range(1,G))
        plot_sim(self.t, self.m, self.p, genes, file)

    def plot_xy(self, g1=1, g2=2, time=False, file=None):
        from ..graphics import plot_xy as plot
        t = np.reshape(self.t, (self.t.size,1))
        stimulus = 1 * (t > 0)
        x = np.append(t, self.m, axis=1)
        y = np.append(stimulus, self.p, axis=1)
        plot(x, y, g1=g1, g2=g2, time=time, file=file)
