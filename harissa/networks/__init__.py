"""A small library for test networks"""


__all__ = ['net0', 'theta_base']
__version__ = '0.1'
__author__ = 'Ulysse Herbach'


from numpy import log
from scipy.special import gammaln

### Utility function
def theta_base(k0, k1, koff, d, m, s):
    """Compute the "neutral value" for theta
    in the case of the auto-activation model."""
    a = k0/d, m, koff/d
    c = (k1-k0)/(d*m)
    theta = a[1]*log(a[2])
    theta += (gammaln(a[0]) - gammaln(a[0] + a[1]*c))/c
    # print(exp(-theta/m)) # Value of s to get theta0 = 0
    return theta + m*log(s) # Correction in the presence of s