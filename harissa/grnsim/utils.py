"""Useful functions for the grnsim package"""
import numpy as np
from scipy.special import gammaln

def theta_base(S0, D0, S1, D1, K0, K1, B, M, S):
    """
    Compute the "neutral value" for theta in the auto-activation model.
    It is consistent in absence of feedback: if M[i,i] = 0, theta[i,i] = 0.
    NB: for this definition to be equivalent with theta_base_stat(a, c)
    of the autoactiv package, one must have the fundamental relations
    K0[i]/D1[i] = a[0,i]
    M[i,i] = a[1,i]
    S[i,i]*B[i]*D0[i]/(S0[i]*S1[i]) = a[2,i]
    (K1[i] - K0[i])/(D1[i]*M[i,i]) = c[i].
    """
    if np.sum(K1-K0 == 0) > 0: return 0*M
    k0, k1, m, s = K0/D1, K1/D1, np.diag(M), np.diag(S)
    theta0 = m * np.log(s*B*D0/(S0*S1))
    theta0 += m * (gammaln(k0) - gammaln(k1)) / (k1 - k0)
    return np.diag(theta0)