"""The class to handle the auto-activation model"""
import numpy as np
from numpy import exp, log
from scipy.special import gamma
from .utils import binom

class model:
    """Parameters of the autoactivation model for a single-gene"""

    def __init__(self, a, theta, c):
        """Store the parameters of the model"""
        self.a = a # a = (a[0],a[1],a[2]) where a[0] = k0, a[1] = m, a[2] = koff/s
        self.theta = theta # Dictionary {timepoint t: theta[t]}
        self.c = c # Cluster number (model = mixture of c+1 gamma distributions)

    def getTimepoints(self):
        """Get the list of time-points in the model"""
        l = list(self.theta)
        l.sort()
        return l

    def getDistribution(self,x,timepoint):
        """Get the distribution associated with the model at a given timepoint:
        - the input is a 1D numpy array of size N representing molecule numbers
        - the distribution is then computed for each value of x
        The output is a (N,T) numpy array where T is the number of time-points."""
        a, theta, c = self.a, self.theta, self.c
        B = binom(c)
        z = np.linspace(0,c,c+1)
        lz = a[0] + a[1]*z
        Z = np.sum(B*exp(theta[timepoint]*z)*gamma(lz)/(a[2]**lz))
        l = (a[0]-1)*log(x) - a[2]*x + c*log(1 + exp(theta[timepoint])*(x**a[1])) - log(Z)
        return exp(l)

    def __repr__(self):
        """How to print model objects"""
        timepoints = self.getTimepoints()
        ltheta = [self.theta[t] for t in timepoints]
        a, c = self.a, self.c
        message = 60*"-"+"\n"
        message += "Autoactivation model ({} time-points):\n".format(len(self.theta))
        message += "k0 = {}, k1 = {},\n".format(a[0],a[0] + c*a[1])
        message += "koff/s = {}, m = {} (c = {})\n".format(a[2], a[1], c)
        message += "Timepoints: {}\n".format(timepoints)
        message += "theta: {}\n".format(ltheta)
        message += 60*"-"
        return message