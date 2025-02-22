"""
Core functions for network inference using likelihood maximization
NB: This is the fast version using Numba
"""
import numpy as np
from numpy import log
from scipy.special import expit, gammaln, psi
from scipy.optimize import minimize
from numba import njit

# Smoothing threshold
s = 0.1

@njit
def p1(x, s):
    """
    Smoothed L1 penalization.
    """
    return (x-s/2)*(x>s) - (x+s/2)*(-x>s) + ((x**2)/(2*s))*(x<=s and -x<=s)

@njit
def grad_p1(x, s):
    """
    Smoothed L1 penalization gradient.
    """
    return 1*(x>s) - 1*(-x>s) + (x/s)*(x<=s and -x<=s)

@njit
def penalization(theta, theta0, t):
    """
    Penalization of network parameters.
    """
    G = theta.shape[0]
    p = 0
    for i in range (1,G):
        # Penalization of basal parameters
        p += 2 * t * p1(theta[i,0]-theta0[i,0], s)
        # Penalization of stimulus parameters
        p += t * p1(theta[0,i]-theta0[0,i], s)
        # Penalization of diagonal parameters
        p += (theta[i,i]-theta0[i,i])**2
        for j in range(1,G):
            # Penalization of interaction parameters
            p += p1(theta[i,j]-theta0[i,j], s)
            if i < j:
                # Competition between interaction parameters
                p += p1(theta[i,j], s) * p1(theta[j,i], s)
    # Final penalization
    return p

@njit
def grad_penalization(theta, theta0, t):
    """
    Penalization gradient of network parameters.
    """
    G = theta.shape[0]
    gradp = np.zeros((G,G))
    for i in range (1,G):
        # Penalization of basal parameters
        gradp[i,0] += 2 * t * grad_p1(theta[i,0]-theta0[i,0], s)
        # Penalization of stimulus parameters
        gradp[0,i] += t * grad_p1(theta[0,i]-theta0[0,i], s)
        # Penalization of diagonal parameters
        gradp[i,i] += 2*(theta[i,i]-theta0[i,i])
        for j in range(1,G):
            # Penalization of interaction parameters
            gradp[i,j] += grad_p1(theta[i,j]-theta0[i,j], s)
            if i != j:
                # Competition between interaction parameters
                gradp[i,j] += grad_p1(theta[i,j], s) * p1(theta[j,i], s)
    # Final penalization
    return gradp

def objective(theta, theta0, x, y, a, c, d, l, t):
    """
    Objective function to be minimized (one time point).
    """
    C, G = x.shape
    theta = theta.reshape((G,G))
    basal = theta[:,0]
    sigma = expit(basal + y @ theta)[:,1:]
    x, y = x[:,1:], y[:,1:]
    a0, a1, c, d = a[0,1:], a[1,1:], c[1:], d[1:]
    # Compute the log-likelihood
    ay, e = a1*y, a0/a1
    cxi = c * (e + (1-e)*sigma)
    q = (d*ay + gammaln(ay + x) - gammaln(ay) - c*y
        + (cxi-1)*log(y) + log(c)*cxi - gammaln(cxi))
    return l*penalization(theta, theta0, t) - np.sum(q)/C

def grad_theta(theta, theta0, x, y, a, c, d, l, t):
    """
    Objective gradient (one time point).
    """
    C, G = x.shape
    theta = theta.reshape((G,G))
    basal = theta[:,0]
    sigma = expit(basal + y @ theta)[:,1:]
    a0, a1, c, d = a[0,1:], a[1,1:], c[1:], d[1:]
    # Pivotal vector u
    e = a0/a1
    xi = e + (1-e)*sigma
    u = c * sigma * (1-xi) * (log(c*y[:,1:]) - psi(c*xi))
    # Compute the objective gradient
    dq = np.zeros((G,G))
    # Basal parameters
    dq[1:,0] += np.sum(u, axis=0)
    # Interaction parameters
    dq[:,1:] += y.T @ u
    dq = l*grad_penalization(theta, theta0, t) - dq/C
    return dq.reshape(G**2)

def infer_proteins(x, a):
    """
    Estimate y directly from data.
    """
    C, G = x.shape
    y = np.ones((C,G))
    z = np.ones((2,G))
    z[0] = a[0]/a[1]
    z[z<1e-5] = 1e-5
    az = a[1]*z
    for k in range(C):
        v = az*log(a[2]/(a[2]+1)) + gammaln(az+x[k]) - gammaln(az)
        for i in range(1,G):
            y[k,i] = z[np.argmax(v[:,i]),i]
    # Stimulus off at t <= 0
    y[x[:,0]<=0,0] = 0
    return y

def infer_network(x, y, a, c, l, tol, verb):
    """
    Network inference procedure.
    """
    C, G = x.shape
    times = np.sort(list(set(x[:,0])))
    T = times.size
    # Useful quantities
    k = x[:,0]
    d = log(a[2]/(a[2]+1))
    # Initialization
    theta = np.zeros((T,G,G))
    theta0 = np.zeros((G,G))
    # Optimization parameters
    params = {'method': 'L-BFGS-B'}
    if tol is not None: params['tol'] = tol
    # Inference routine
    for t, time in enumerate(times):
        res = minimize(objective, theta0.reshape(G**2),
                args=(theta0, x[k==time], y[k==time], a, c, d, l, t),
                jac=grad_theta, **params)
        if not res.success:
            print(f'Warning: maximization failed (time {t})')
        # Update theta0
        theta0 = res.x.reshape((G,G))
        # Store theta at time t
        theta[t] = theta0
    if verb: print(f'Fitted theta in {res.nit} iterations')
    return theta
