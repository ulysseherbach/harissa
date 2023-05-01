"""
Store simulations
"""

class Simulation:
    """
    Basic object to store simulations.
    """
    def __init__(self, t, m, p):
        self.t = t # Time points
        self.m = m # mRNAs
        self.p = p # Proteins
