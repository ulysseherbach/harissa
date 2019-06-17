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
        Note that array inter[t] can be either dense or sparse (csc format).
    basal : array (G,)
        Basal activity for each gene.
    a : array (G,)
        Relative parameter k1/d0 for each gene.
    b : array (G,)
        Relative parameter koff/s0 for each gene.
    c : array (G,)
        Relative parameter k1/d1 for each gene.
    mask : array (G, G)
        mask[i,j] = 1 if interaction i -> j is possible and 0 otherwise.
        This array can be either dense or sparse (csc format).
"""
import numpy as np
from numpy import exp, log
from scipy.special import gammaln, psi
from scipy import sparse
from scipy.optimize import minimize

def penalization(inter, l):
    """
    Penalization of interaction parameters.
    """
    if l == 0: return 0
    p = 0
    for t in inter.keys():
        if not sparse.issparse(inter[t]): p += np.sum(inter[t]**2)
        else: p += inter[t].multiply(inter[t]).sum()
    return l * p

def obj_cell(x, y, inter, basal, a, c, d):
    """
    Objective function to be maximized (one cell).
    """
    phi = exp(basal + y @ inter)
    # Remove the stimulus
    sigma = phi[1:] / (phi[1:] + 1)
    x, y = x[1:], y[1:]
    a, c, d = a[1:], c[1:], d[1:]
    # Compute the log-likelihood
    asigma = a * sigma
    csigma = c * sigma
    q = (d*asigma + gammaln(asigma + x) - gammaln(asigma) - c*y
        + (csigma-1)*log(y) + log(c)*csigma - gammaln(csigma))
    return np.sum(q)

def obj_t(x, y, inter, basal, a, c, d):
    """
    Objective function to be maximized (all cells at one time point).
    """
    phi = exp(basal + y @ inter)
    # Remove the stimulus
    sigma = phi[:,1:] / (phi[:,1:] + 1)
    x, y = x[:,1:], y[:,1:]
    a, c, d = a[1:], c[1:], d[1:]
    # Compute the log-likelihood
    asigma = a * sigma
    csigma = c * sigma
    q = (d*asigma + gammaln(asigma + x) - gammaln(asigma) - c*y
        + (csigma-1)*log(y) + log(c)*csigma - gammaln(csigma))
    return np.sum(q)

def objective(x, y, inter, basal, a, b, c, l):
    """
    Objective function to be maximized (all cells).
    """
    C, G = x.shape
    t = x[:,0]
    times = set(t)
    d = log(b/(b+1))
    Q = 0
    for time in times:
        Q += obj_t(x[t==time], y[t==time], inter[time], basal, a, c, d)
    return Q/C - penalization(inter, l)

def grad_y_obj_cell(x, y, inter, basal, a, c, d):
    """
    Objective function gradient with respect to y (one cell).
    """
    phi = exp(basal + y @ inter)
    # Remove the stimulus
    sigma = phi[1:] / (phi[1:] + 1)
    # if np.min(sigma) == 0: print('Warning: sigma = 0')
    x, y = x[1:], y[1:]
    a, c, d = a[1:], c[1:], d[1:]
    # Compute the log-likelihood gradient
    asigma = a * sigma
    csigma = c * sigma
    u = sigma * (1-sigma) * (c*(log(c*y) - psi(csigma))
        + a*(d + psi(asigma+x) - psi(asigma)))
    v = (csigma-1)/y - c
    # Restore the stimulus
    u, v = np.append(0,u), np.append(0,v)
    dq = inter @ u + v
    dq[0] = 0
    return dq

# def get_u_cell(x, y, inter, basal, a, c, d):
#     """
#     Compute the pivotal vector u for one cell.
#     """
#     phi = exp(basal + y @ inter)
#     # Remove the stimulus
#     sigma = phi[1:] / (phi[1:] + 1)
#     x, y = x[1:], y[1:]
#     a, c, d = a[1:], c[1:], d[1:]
#     # Compute the log-likelihood
#     asigma = a * sigma
#     csigma = c * sigma
#     u = sigma * (1-sigma) * (c*(log(c*y) - psi(csigma))
#         + a*(d + psi(asigma+x) - psi(asigma)))
#     return np.append(0,u)

def get_u_t(x, y, inter, basal, a, c, d):
    """
    Compute the pivotal vector u for all cells at one time point.
    """
    phi = exp(basal + y @ inter)
    # Remove the stimulus
    sigma = phi[:,1:] / (phi[:,1:] + 1)
    x, y = x[:,1:], y[:,1:]
    a, c, d = a[1:], c[1:], d[1:]
    # Compute the log-likelihood
    asigma = a * sigma
    csigma = c * sigma
    u = sigma * (1-sigma) * (c*(log(c*y) - psi(csigma))
        + a*(d + psi(asigma+x) - psi(asigma)))
    return np.append(np.zeros((x[:,0].size,1)), u, axis=1)

# def grad_theta_obj(x, y, inter, basal, a, c, d, mask, l):
#     """
#     Objective gradient with respect to theta = (basal, inter) for all cells.
#     The result is in the form dq = (dbasal, dinter) with dinter[t] being
#     either dense if inter[t] is dense or sparse if inter[t] is sparse.
#     """
#     C, G = x.shape
#     t = x[:,0]
#     # Initialize dbasal and dinter
#     dbasal = np.zeros(G)
#     dinter = {}
#     # Add terms while keeping the sparsity structure
#     for k in range(C):
#         u = get_u_cell(x[k], y[k], inter[t[k]], basal, a, c, d)
#         dbasal += u
#         # Build dinter for cell k
#         if mask is None:
#             v = np.diag(y[k]) @ np.ones((G,G)) @ np.diag(u)
#         elif not sparse.issparse(mask):
#             v = np.diag(y[k]) @ abs(mask) @ np.diag(u)
#         else:
#             v = sparse.diags(y[k]) * abs(mask) * sparse.diags(u)
#         # Add it to the relevant time point
#         if t[k] in dinter: dinter[t[k]] += v
#         else: dinter[t[k]] = v
#     dbasal = dbasal/C
#     for t in dinter.keys(): dinter[t] = dinter[t]/C - 2*l*inter[t]
#     return dbasal, dinter

def grad_theta_obj(x, y, inter, basal, a, c, d, mask, l):
    """
    Objective gradient with respect to theta = (basal, inter) for all cells.
    The result is in the form dq = (dbasal, dinter) with dinter[t] being
    either dense if inter[t] is dense or sparse if inter[t] is sparse.
    """
    C, G = x.shape
    t = x[:,0]
    times = set(t)
    # Initialize dbasal and dinter
    dbasal = np.zeros(G)
    dinter = {}
    # Add terms while keeping the sparsity structure
    for time in times:
        u = get_u_t(x[t==time], y[t==time], inter[time], basal, a, c, d)
        dbasal += np.sum(u, axis=0)
        # Build dinter for all cells at given time point
        m = y[t==time].T @ u
        if mask is None: dinter[time] = m
        elif not sparse.issparse(mask): dinter[time] = abs(mask) * m
        else: dinter[time] = abs(mask).multiply(m)
        dinter[time] = dinter[time]/C - 2*l*inter[time]
    dbasal = dbasal/C
    return dbasal, dinter

def expectation(x, y, inter, basal, a, c, d, verb=False):
    """
    Approximate expectation step.
    """
    C, G = x.shape
    t = x[:,0]
    n = 0
    # Optimization of p = y[k] for each cell k
    for k in range(C):
        # print(exp(basal + y[k] @ inter[t[k]]))
        def f(p):
            return -obj_cell(x[k], p, inter[t[k]], basal, a, c, d)
        def Df(p):
            return -grad_y_obj_cell(x[k], p, inter[t[k]], basal, a, c, d)
        p0 = y[k]
        bounds = [(None,None)] + (G-1) * [(1e-5,None)]
        res = minimize(f, p0, method='L-BFGS-B', jac=Df, bounds=bounds)
        # if not res.success: print('Warning, expectation step failed')
        y[k] = res.x
        # if verb: print('Fitted y[{}] in {} iterations'.format(k+1,res.nit))
        n += res.nit
    if verb: print('Fitted y in {:.2f} iterations on average'.format(n/C))

def maximization(x, y, inter, basal, a, b, c, mask, l, verb=False):
    """
    Maximization step.
    """
    C, G = x.shape
    times = list(set(x[:,0]))
    T = len(times)
    d = log(b/(b+1))

    # Dense case
    if not sparse.issparse(mask):
        N = G**2
        # Build a 1-D array to store parameters
        theta0 = np.zeros(G + T*N)
        theta0[0:G] = basal
        for k, t in enumerate(times):
            theta0[G+k*N:G+(k+1)*N] = np.reshape(inter[t], (N,))
        # Define optimization functions
        def f(theta):
            basal0 = theta[0:G]
            inter0 = {}
            for k, t in enumerate(times):
                inter0[t] = np.reshape(theta[G+k*N:G+(k+1)*N], (G,G))
            return -objective(x, y, inter0, basal0, a, b, c, l)
        def Df(theta):
            basal0 = theta[0:G]
            inter0 = {}
            for k, t in enumerate(times):
                inter0[t] = np.reshape(theta[G+k*N:G+(k+1)*N], (G,G))
            dq = grad_theta_obj(x, y, inter0, basal0, a, c, d, mask, l)
            dtheta = np.zeros(G + T*N)
            dtheta[0:G] = -dq[0]
            for k, t in enumerate(times):
                dtheta[G+k*N:G+(k+1)*N] = np.reshape(-dq[1][t], (N,))
            return dtheta
        if mask is not None: vmask = np.reshape(mask, (N,))
        else: vmask = np.zeros(N)

    # Sparse case
    else:
        I, J, V = sparse.find(mask)
        N = V.size
        # Build a 1-D array to store parameters
        theta0 = np.zeros(G + T*N)
        theta0[0:G] = basal
        for k, t in enumerate(times):
            theta0[G+k*N:G+(k+1)*N] = inter[t][I,J]
        # Define optimization functions
        def f(theta):
            basal0 = theta[0:G]
            inter0 = {}
            for k, t in enumerate(times):
                Vtheta = theta[G+k*N:G+(k+1)*N]
                inter0[t] = sparse.csc_matrix((Vtheta,(I,J)), (G,G))
            return -objective(x, y, inter0, basal0, a, b, c, l)
        def Df(theta):
            basal0 = theta[0:G]
            inter0 = {}
            for k, t in enumerate(times):
                Vtheta = theta[G+k*N:G+(k+1)*N]
                inter0[t] = sparse.csc_matrix((Vtheta,(I,J)), (G,G))
            dq = grad_theta_obj(x, y, inter0, basal0, a, c, d, mask, l)
            dtheta = np.zeros(G + T*N)
            dtheta[0:G] = -dq[0]
            for k, t in enumerate(times):
                dtheta[G+k*N:G+(k+1)*N] = -dq[1][t][I,J]
            return dtheta
        vmask = V

    # Solve the minimization problem
    # interb = []
    # for i in range(N):
    #     if vmask[i] > 0: interb.append((0,None))
    #     elif vmask[i] < 0: interb.append((None,0))
    #     else: interb.append((None,None))
    # bounds = G * [(None,None)] + T * interb
    bounds = (G + T*N) * [(None,None)]

    res = minimize(f, theta0, method='L-BFGS-B', jac=Df, bounds=bounds)
    theta = res.x
    basal[:] = theta[0:G]
    for k, t in enumerate(times):
        Vtheta = theta[G+k*N:G+(k+1)*N]
        if not sparse.issparse(mask): inter[t] = np.reshape(Vtheta, (G,G))
        else: inter[t] = sparse.csc_matrix((Vtheta,(I,J)), (G,G))
    if not res.success: print('Warning, maximization step failed')
    if verb: print('Fitted theta in {} iterations'.format(res.nit))

def inference(x, inter, basal, a, b, c, mask, l=1e-3,
    tol=1e-4, max_iter=10, verb=False):
    """
    Network inference procedure using a CEM algorithm.
    Return the list of successive objective function values.
    """
    C, G = x.shape
    time = x[:,0]
    d = log(b/(b+1))
    y = np.ones((C,G))
    y[time<=0,0] = 0 # Stimulus at t <= 0
    # Initialization
    if verb: print('EM initialization...')
    q = [objective(x, y, inter, basal, a, b, c, l)]
    obj_increase = tol + 1
    iter_count = 0
    while (iter_count < max_iter) and (obj_increase > tol):
        print('EM iteration {}...'.format(iter_count+1))
        # EM routine
        expectation(x, y, inter, basal, a, c, d, verb=verb)
        maximization(x, y, inter, basal, a, b, c, mask, l, verb=verb)
        obj_new = objective(x, y, inter, basal, a, b, c, l)
        q.append(obj_new)
        obj_increase = q[-1] - q[-2]
        iter_count += 1
    return y, q
