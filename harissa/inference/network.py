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

# Fused penalization strength
lf = 1

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
    a, c, d = a[1:], c[1:], d[1:]
    # Compute the log-likelihood
    ay = a * y
    csigma = c * sigma
    q = (d*ay + gammaln(ay + x) - gammaln(ay) - c*y
        + (csigma-1)*log(y) + log(c)*csigma - gammaln(csigma))
    return np.sum(q)

def obj_t(x, y, inter, basal, a, c, d):
    """
    Basic objective function (all cells at one time point).
    """
    # Remove the stimulus
    sigma = expit(basal + y @ inter)[:,1:]
    x, y = x[:,1:], y[:,1:]
    a, c, d = a[1:], c[1:], d[1:]
    # Compute the log-likelihood
    ay = a * y
    csigma = c * sigma
    q = (d*ay + gammaln(ay + x) - gammaln(ay) - c*y
        + (csigma-1)*log(y) + log(c)*csigma - gammaln(csigma))
    return np.sum(q)

def objective(x, y, inter, basal, a, b, c, l):
    """
    Objective function to be maximized (all cells).
    """
    t = x[:,0]
    C = x[:,0].size
    times = list(inter.keys())
    times.sort()
    d = log(b/(b+1))
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

def grad_y_obj_cell(x, y, inter, basal, a, c, d):
    """
    Objective function gradient with respect to y (one cell).
    """
    # Remove the stimulus
    sigma = expit(basal + y @ inter)[1:]
    x, y = x[1:], y[1:]
    a, c, d = a[1:], c[1:], d[1:]
    # if np.min(sigma) == 0: print('Warning: sigma = 0')
    # Compute the log-likelihood gradient
    ay = a * y
    csigma = c * sigma
    # Take care of stabilizing the digamma function
    u = (1-sigma) * (csigma*log(c*y) - csigma*psi(csigma+1) + 1)
    v = (csigma-1)/y - c + a*(d + psi(ay+x) - psi(ay))
    # Restore the stimulus
    u, v = np.append(0,u), np.append(0,v)
    dq = inter @ u + v
    dq[0] = 0
    return dq

def u_t_gene(i, y, inter, basal, c):
    """
    Compute the pivotal vector u for gene i for all cells at one time point.
    """
    sigma = expit(basal + y @ inter)
    # Compute the log-likelihood
    csigma = c * sigma
    # Take care of stabilizing the digamma function
    u = (1-sigma) * (csigma*log(c*y[:,i]) - csigma*psi(csigma+1) + 1)
    return u

def grad_theta_gene(i, x, y, inter, basal, c, mask, l):
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
        u = u_t_gene(i, y[t==time], inter[time], basal, c)
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
    # Optimization of p = y[k] for each cell k
    for k in range(C):
        def f(p):
            return -obj_cell(x[k], p, inter[t[k]], basal, a, c, d)
        def Df(p):
            return -grad_y_obj_cell(x[k], p, inter[t[k]], basal, a, c, d)
        p0 = y[k]
        bounds = [(None,None)] + (G-1) * [(1e-5,None)]
        # options = {'gtol': 1e-1}
        res = minimize(f, p0, method='L-BFGS-B', jac=Df, bounds=bounds,
            tol=1e-4)
        if not res.success: print('Warning, expectation step failed')
        y[k] = res.x
        # if verb: print('Fitted y[{}] in {} iterations'.format(k+1,res.nit))
        n += res.nit
    # if verb: print('\tFitted y (avg.) {:.2f} iterations'.format(n/C))
    if verb: print('\tmax[y] = {}'.format(np.max(y)))

def maximization(x, y, inter, basal, a, b, c, mask, sign, l, verb=False):
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
            dq = grad_theta_gene(i, x, y, inter0, basal0, c[i], mask[:,i], l)
            dtheta = np.zeros(1 + T*G)
            dtheta[0] = -dq[0]
            for k, t in enumerate(times):
                dtheta[1+k*G:1+(k+1)*G] = -dq[1][t]
            return dtheta
        # Possible sign constraints
        bounds = None
        if sign is not None:
            bounds = (1 + T*G) * [(None,None)]
            for k in range(T):
                for j in range(G):
                    if sign[j,i] > 0: bounds[1+k*G+j] = (0,None)
                    if sign[j,i] < 0: bounds[1+k*G+j] = (None,0)
        # Solve the minimization problem
        # options = {'gtol': 1e-2}
        res = minimize(f, theta0, method='L-BFGS-B', jac=Df, bounds=bounds,
            tol=1e-4)
        theta = res.x
        basal[i] = theta[0]
        for k, t in enumerate(times): inter[t][:,i] = theta[1+k*G:1+(k+1)*G]
        # if not res.success: print('Warning, maximization step failed')
        n += res.nit
        # if verb: print('Fitted gene {} in {} iterations'.format(i, res.nit))
    # if verb: print('\tFitted inter (avg.) {:.2f} iterations'.format(n/(G-1)))

def inference(x, inter, basal, a, b, c, l, tol, mask, sign, max_iter,
    save, verb):
    """
    Network inference procedure using a CEM algorithm.
    Return the list of successive objective function values.
    """
    C, G = x.shape
    times = list(set(x[:,0]))
    times.sort()
    T = len(times)
    d = log(b/(b+1))
    y = np.ones((C,G))
    y[x[:,0]<=0,0] = 0 # Stimulus at t <= 0
    # Mask
    if mask is None: mask = np.ones((G,G),dtype='int') - np.eye(G,dtype='int')
    mask[:,0] = 0
    # Initialization
    if verb: print('EM initialization...')
    q = [objective(x, y, inter, basal, a, b, c, l)]
    obj_increase = tol + 1
    iter_count = 0
    if G == 1:
        print('There is nothing to infer!')
        iter_count = max_iter
    while (iter_count < max_iter) and (obj_increase > tol):
        if verb: print('EM iteration {}...'.format(iter_count+1))
        # EM routine
        expectation(x, y, inter, basal, a, c, d, verb=verb)
        maximization(x, y, inter, basal, a, b, c, mask, sign, l, verb=verb)
        obj_new = objective(x, y, inter, basal, a, b, c, l)
        q.append(obj_new)
        # obj_increase = q[-1] - q[-2]
        obj_increase = (q[-1] - q[-2])/max([abs(q[-2]),abs(q[-1]),1])
        iter_count += 1
        # Verbose
        if verb:
            maxinter = np.max([np.max(np.abs(inter[t])) for t in times])
            print('\tmax|inter| = {}'.format(maxinter))
            print('\tObjective: {}'.format(obj_new))
        # Saving
        if save is not None:
            np.save(save+'/basal', basal)
            inter_ = np.zeros((T,G,G))
            for k, t in enumerate(times): inter_[k] = inter[t]
            np.save(save+'/inter', inter_)
    if iter_count < max_iter:
        print('Converged ({} steps)'.format(iter_count))
        # if verb: print('Converged ({} steps)'.format(iter_count))
    else: print('Warning: not converged in {} iterations'.format(max_iter))
    return y, q
