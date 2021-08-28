"""
Core functions for network inference using the CEM algorithm.
See documentation for details on the statistical model.

Considered sizes
----------------
    C : number of cells
    G : number of genes including stimulus

Parameters
----------
    x : array (C, G)
        Observed mRNA levels (column 0 = time points).
    y : array (C, G)
        Latent protein levels (column 0 = stimulus).
    inter : dictionnary of arrays (G, G)
        Given key t, inter[t][i,j] denotes interaction i -> j at time t.
    basal : array (G,)
        Basal activity for each gene.
    a : array (G,)
        Relative parameter k1/d0 for each gene.
    b : array (G,)
        Relative parameter koff/s0 for each gene.
    c : array (G,)
        Relative parameter k1/d1 for each gene.
    mask : array (G, G)
        mask[i,j] = +1 (resp. -1) if i -> j (resp. i -| j), otherwise 0.
"""
import numpy as np
from numpy import log
from scipy.special import gammaln, psi
from scipy.optimize import minimize
from scipy.special import expit

# Tolerance parameter for the EM steps
em_tol = 1e-4

# Fused penalization strength
# lf = 1

def penalization(inter, times, l):
    """
    Ridge penalization of interaction parameters.
    """
    for k, time in enumerate(times):
        if k == 0: p = 0.5 * inter[time]**2
        else: p += 0.5 * inter[time]**2
    # # OPTION: fused ridge penalization
    # dt = np.array(times)
    # dt = dt[1:] - dt[:-1]
    # dt = dt/(lf*np.min(dt))
    # for k in range(len(times)-1):
    #     t, t1 = times[k], times[k+1]
    #     p += (0.5/dt[k]) * (inter[t1] - inter[t])**2
    return l * np.sum(p)

def grad_penalization(inter, times, l):
    """
    Ridge penalization gradient of interaction parameters.
    """
    gradp = {t: inter[t].copy() for t in times}
    # # OPTION: fused ridge penalization
    # dt = np.array(times)
    # dt = dt[1:] - dt[:-1]
    # dt = dt/(lf*np.min(dt))
    # for k in range(len(times)):
    #     if k == 0:
    #         t, t1 = times[0], times[1]
    #         gradp[t] += (inter[t] - inter[t1])/dt[k]
    #     elif k < len(times)-1:
    #         t0, t, t1 = times[k-1], times[k], times[k+1]
    #         gradp[t] += (inter[t] - inter[t0])/dt[k-1]
    #         gradp[t] += (inter[t] - inter[t1])/dt[k]
    #     else:
    #         t0, t = times[k-1], times[k]
    #         gradp[t] += (inter[t] - inter[t0])/dt[k-1]
    return {t: l * gradp[t] for t in times}

def obj_cell(x, y, inter, basal, a, c, d):
    """
    Objective function to be maximized in y (one cell).
    """
    # Remove the stimulus
    sigma = expit(basal + y @ inter)[1:]
    x, y = x[1:], y[1:]
    a0, a1, c, d = a[0,1:], a[1,1:], c[1:], d[1:]
    # Compute the log-likelihood
    ay = a1 * y
    e = a0/a1
    cxi = c * (e + (1-e)*sigma)
    q = (d*ay + gammaln(ay + x) - gammaln(ay) - c*y
        + (cxi-1)*log(y) + log(c)*cxi - gammaln(cxi))
    return np.sum(q)

def obj_t(x, y, inter, basal, a, c, d):
    """
    Basic objective function (all cells at one time point).
    """
    # Remove the stimulus
    sigma = expit(basal + y @ inter)[:,1:]
    x, y = x[:,1:], y[:,1:]
    a0, a1, c, d = a[0,1:], a[1,1:], c[1:], d[1:]
    # Compute the log-likelihood
    ay = a1 * y
    e = a0/a1
    cxi = c * (e + (1-e)*sigma)
    q = (d*ay + gammaln(ay + x) - gammaln(ay) - c*y
        + (cxi-1)*log(y) + log(c)*cxi - gammaln(cxi))
    return np.sum(q)

def objective(x, y, inter, basal, a, c, d, l):
    """
    Objective function to be maximized (all cells).
    """
    t = x[:,0]
    C = x[:,0].size
    times = list(inter.keys())
    times.sort()
    Q = 0
    for time in times:
        Q += obj_t(x[t==time], y[t==time], inter[time], basal, a, c, d)
    # Compute penalization
    Q -= penalization(inter, times, l)
    return Q/C

def obj_t_gene(i, y, inter, basal, c):
    """
    Objective function for gene i (all cells at one time point).
    """
    sigma = expit(basal + y @ inter)
    # Compute the log-likelihood
    csigma = c * sigma
    q = (csigma-1)*log(y[:,i]) + log(c)*csigma - gammaln(csigma)
    return np.sum(q)

def obj_gene(i, x, y, inter, basal, c, l):
    """
    Objective function for gene i (all cells).
    """
    t = x[:,0]
    C = x[:,0].size
    times = list(inter.keys())
    times.sort()
    Q = 0
    for time in times:
        Q += obj_t_gene(i, y[t==time], inter[time], basal, c)
    # Compute penalization
    Q -= penalization(inter, times, l)
    return Q/C

def u_t_gene(i, y, inter, basal, a, c):
    """
    Compute the pivotal vector u for gene i for all cells at one time point.
    """
    sigma = expit(basal + y @ inter)
    # Compute the log-likelihood
    e = a[0]/a[1]
    xi = e + (1-e)*sigma
    # Take care of stabilizing the digamma function
    u = c * sigma * (1-xi) * (log(c*y[:,i]) - psi(c*xi))
    return u

def grad_theta_gene(i, x, y, inter, basal, a, c, mask, l):
    """
    Objective gradient for gene i for all cells.
    """
    t = x[:,0]
    C = x[:,0].size
    times = list(inter.keys())
    times.sort()
    # Initialize dbasal and dinter
    dbasal = 0
    dinter = {}
    # Compute penalization gradient
    p = grad_penalization(inter, times, l)
    # Add terms while keeping the sparsity structure
    for time in times:
        u = u_t_gene(i, y[t==time], inter[time], basal, a, c)
        dbasal += np.sum(u)
        dinter[time] = abs(mask) * (y[t==time].T @ u)
    # # OPTION: no interactions at t = 0
    # dinter[0] = 0
    # OPTION: same inter[0,:] at each time
    inter0 = np.sum([dinter[time][0] for time in times])
    for time in times: dinter[time][0] = inter0
    # Add penalization gradient
    for time in times: dinter[time] -= p[time]
    return dbasal/C, {time: dinter[time]/C for time in times}

def expectation(x, y, inter, basal, a, c, d, verb=False):
    """
    Approximate expectation step.
    """
    C, G = x.shape
    t = x[:,0]
    n = 0
    # Discrete optimization of y[k] for each cell k
    # for k in range(C):

    #     if not res.success: print('Warning, expectation step failed')
    #     y[k] = res.x
    #     # if verb: print('Fitted y[{}] in {} iterations'.format(k+1,res.nit))
    #     n += res.nit
    # if verb: print('\tFitted y (avg.) {:.2f} iterations'.format(n/C))
    if verb: print('\tmax[y] = {}'.format(np.max(y)))

def maximization(x, y, inter, basal, a, c, mask, l, verb=False):
    """
    Maximization step.
    """
    C, G = x.shape
    times = list(set(x[:,0]))
    T = len(times)
    n = 0
    for i in range(1,G):
        # Build a 1-D array to store parameters
        theta0 = np.zeros(1 + T*G)
        theta0[0] = basal[i]
        for k, t in enumerate(times): theta0[1+k*G:1+(k+1)*G] = inter[t][:,i]
        # Define optimization functions
        def f(theta):
            basal0 = theta[0]
            inter0 = {t: theta[1+k*G:1+(k+1)*G] for k, t in enumerate(times)}
            return -obj_gene(i, x, y, inter0, basal0, c[i], l)
        def Df(theta):
            basal0 = theta[0]
            inter0 = {t: theta[1+k*G:1+(k+1)*G] for k, t in enumerate(times)}
            dq = grad_theta_gene(i, x, y, inter0, basal0, a[:,i], c[i], mask[:,i], l)
            dtheta = np.zeros(1 + T*G)
            dtheta[0] = -dq[0]
            for k, t in enumerate(times):
                dtheta[1+k*G:1+(k+1)*G] = -dq[1][t]
            return dtheta
        # Possible sign constraints
        bounds = None
        # Solve the minimization problem
        # options = {'gtol': 1e-2}
        res = minimize(f, theta0, method='L-BFGS-B', jac=Df, bounds=bounds,
            tol=em_tol)
        theta = res.x
        basal[i] = theta[0]
        for k, t in enumerate(times): inter[t][:,i] = theta[1+k*G:1+(k+1)*G]
        # if not res.success: print('Warning, maximization step failed')
        n += res.nit
        # if verb: print('Fitted gene {} in {} iterations'.format(i, res.nit))
    # if verb: print('\tFitted inter (avg.) {:.2f} iterations'.format(n/(G-1)))

def inference(x, inter, basal, a, l, tol, max_iter, verb):
    """
    Network inference procedure using a CEM algorithm.
    Return the list of successive objective function values.
    """
    C, G = x.shape
    times = np.sort(list(set(x[:,0])))
    # Useful parameters
    c = 10 * np.ones(G)
    d = log(a[2]/(a[2]+1))
    y = np.ones((C,G))
    y[x[:,0]<=0,0] = 0 # Stimulus at t <= 0
    # Mask
    mask = np.ones((G,G),dtype='int') - np.eye(G,dtype='int')
    mask[:,0] = 0
    # Initialization
    if verb: print('EM initialization...')
    q = [objective(x, y, inter, basal, a, c, d, l)]
    obj_increase = tol + 1
    iter_count = 0
    if G == 1:
        print('There is nothing to infer!')
        iter_count = max_iter
    while (iter_count < max_iter) and (obj_increase > tol):
        if verb: print('EM iteration {}...'.format(iter_count+1))
        # EM routine
        expectation(x, y, inter, basal, a, c, d, verb=verb)
        maximization(x, y, inter, basal, a, c, mask, l, verb=verb)
        obj_new = objective(x, y, inter, basal, a, c, d, l)
        q.append(obj_new)
        # obj_increase = q[-1] - q[-2]
        obj_increase = (q[-1] - q[-2])/max([abs(q[-2]),abs(q[-1]),1])
        iter_count += 1
        # Verbose
        if verb:
            maxinter = np.max([np.max(np.abs(inter[t])) for t in times])
            print('\tmax|inter| = {}'.format(maxinter))
            print('\tObjective: {}'.format(obj_new))
    if iter_count < max_iter:
        print('Converged ({} steps)'.format(iter_count))
        # if verb: print('Converged ({} steps)'.format(iter_count))
    else: print('Warning: not converged in {} iterations'.format(max_iter))
    return y, q
