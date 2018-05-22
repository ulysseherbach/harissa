"""Core functions for inferring the gamma-binomial auto-model"""
import numpy as np
from . import config as cf
from . import utils as ut
from . import sumproduct as sp
from numpy import exp, log
from scipy.special import gammaln
import scipy.optimize

def gibbs_sample(x, z, iter_gibbs, a, theta, c):
    """Gibbs sampling of X and Z"""
    K, G = np.shape(x)
    for k in range(K):
        X, Z = x[k], z[k] # Views
        for t in range(iter_gibbs):
            ### Phase 1: sample X knowing Z
            for i in range(G):
                X[i] = np.random.gamma(a[0,i] + a[1,i]*Z[i], 1/a[2,i])
            ### Phase 2: sample Z knowing X
            rsort = np.random.permutation(G) # Random sorting of genes
            for i in rsort:
                Z[i] = 1
                w = exp(np.sum(theta[i,:]*Z)) * (X[i]**a[1,i])
                Z[i] = np.random.binomial(c[i], w/(1+w))

def meanz_var_cell(alpha, a, c, x):
    """Mean of the variational distribution of z for a cell x"""
    G = np.size(alpha)
    m = c*exp(alpha)/(1+exp(alpha))
    for i in range(G):
        ### Modifiy the expectations for missing data
        if not (x[i] > 0):
            z = np.array(range(c[i]+1))
            v = ut.binomln(c[i]) + alpha[i]*z + gammaln(a[0,i]+a[1,i]*z)
            p = exp(v-np.max(v)) # Trick to keep small values
            p = p/np.sum(p)
            m[i] = np.sum(z*p)
    return m

def meanz_var(alpha, a, c, data):
    """Compute the mean matrix of the variational distribution of z"""
    C, G = np.shape(alpha)
    E = np.zeros((G,G))
    for k in range(C):
        Alpha, x = alpha[k], data[k] # Views
        m = c*exp(Alpha)/(1+exp(Alpha))
        for i in range(G):
            ### Modifiy the expectations for missing data
            if not (x[i] > 0):
                z = np.array(range(c[i]+1))
                v = ut.binomln(c[i]) + Alpha[i]*z + gammaln(a[0,i]+a[1,i]*z)
                p = exp(v-np.max(v)) # Trick to keep small values
                p = p/np.sum(p)
                m[i] = np.sum(z*p)
            ### Take the expectations
            E[i,i] += m[i]
            for j in range(i+1,G):
                E[i,j] += m[i] * m[j]
    ### Option: symmetrize E
    for i in range(G):
        for j in range(i+1,G):
            E[j,i] = E[i,j]
    return E/C

def e_step_var(alpha, data, a, theta, c, info=True):
    """Perform E step using the variational method"""
    tol = cf.var_tol
    iter_max = cf.var_iter_max
    C, G = np.shape(alpha)
    for k in range(C):
        ### Get the variational parameters for each cell
        Alpha, x = alpha[k], data[k] # Views
        count, v = 0, tol + 1
        while ((v > tol) & (count < iter_max)):
            Alpha0 = Alpha.copy()
            rsort = np.random.permutation(G) # Random sorting of genes
            # for i in range(G):
            for i in rsort:
                w = meanz_var_cell(Alpha, a, c, x)
                w[i] = 1
                if (x[i] > 0):
                    Alpha[i] = a[1,i]*log(x[i]) + np.sum(theta[i]*w)
                else:
                    Alpha[i] = a[1,i]*log(a[2,i]) + np.sum(theta[i]*w)
            v = np.max(np.abs(Alpha - Alpha0))
            count += 1
        # print(k)
        if (count == iter_max):
            print('Warning: bad convergence of the variational E step')
    if info: print('E step done')
    return alpha

def m_step_mc(x, z, alpha, data, a, theta, c):
    """Perform the M step by Monte Carlo"""
    ngibbs = cf.iter_gibbs
    ngrad = cf.iter_grad
    tau = cf.learning_rate
    pen = cf.penalization
    mix = cf.lasso_mix
    G = np.size(c)
    E = meanz_var(alpha, a, c, data)
    ### Perform basic gradient method
    for k in range(ngrad):
        ### Resample X and Z
        gibbs_sample(x, z, ngibbs, a, theta, c)
        ### Estimate the gradient
        Dtheta = np.zeros((G,G))
        for i in range(G):
            Dtheta[i,i] = E[i,i] - np.mean(z[:,i])
            for j in range(i+1,G):
                Dtheta[i,j] = E[i,j] - np.mean(z[:,i]*z[:,j])
                Dtheta[j,i] = Dtheta[i,j]
        theta = ut.prox_operator(theta + tau*Dtheta, tau, pen, mix)
    print('M step done')
    return theta

def m_step_brute(states, alpha, data, a, theta, c, info=True):
    """Perform the M step by full sum (small networks)"""
    ngrad = cf.iter_grad
    tau = cf.learning_rate
    mix = cf.lasso_mix
    tol = cf.m_tol
    G = np.size(c)
    E = meanz_var(alpha, a, c, data)
    S = ut.zscales(c) # Scaling constant for theta entries
    ### Perform basic gradient method
    t = 0
    v = tol + 1 # Check variation of theta
    v0 = v
    while ((t < ngrad) & (v > tol)):
        t += 1
        M = ut.meanz_brute(states, a, theta, c)
        Dtheta = np.zeros((G,G))
        for i in range(G):
            Dtheta[i,i] = E[i,i] - M[i,i]
            for j in range(i+1,G):
                Dtheta[i,j] = E[i,j] - M[i,j]
                Dtheta[j,i] = Dtheta[i,j]
        ### Perform gradient step
        theta0 = theta
        rate = tau * S # Normalized learning rate
        pen = cf.penalization * S
        theta = ut.prox_operator(theta + rate*Dtheta, rate, pen, mix)
        v = np.max(np.abs(theta - theta0))
        # v = np.sqrt(np.sum((theta - theta0)**2))
        # v = np.sqrt(np.sum(Dtheta**2)) # Only ok if no penalization
        # print(v)
        if not v < v0:
            tau *= 0.5
            # cf.learning_rate = tau
            # print('learning rate -> {:.2e}'.format(tau))
        v0 = v
    if info: print('M step done in {} iterations\n'.format(t))
    return theta

def objQ(states, alpha, data, a, theta, c):
    """Objective function to be maximized in alpha and theta"""
    C, G = np.shape(alpha)
    Q = 0
    for k in range(C):
        Alpha, x = alpha[k], data[k] # Views
        m = meanz_var_cell(Alpha, a, c, x)
        for i in range(G):
            if (x[i] > 0):
                Q += (theta[i,i] + a[1,i]*log(x[i]) - Alpha[i]) * m[i]
                Q += c[i]*log(1 + exp(Alpha[i]))
            else:
                Q += (theta[i,i] + a[1,i]*log(a[2,i]) - Alpha[i]) * m[i]
                z = np.array(range(c[i]+1))
                v = ut.binomln(c[i]) + Alpha[i]*z + gammaln(a[0,i]+a[1,i]*z)
                Q += log(np.sum(exp(v)))
            for j in range(i+1,G):
                Q += theta[i,j] * m[i] * m[j]
    return Q/C - log(ut.cst_brute(states, a, theta, c))

def m_step_bf_qn(states, alpha, data, a, theta, c, info=True):
    """Perform the M step by full sum and a quasi-newton method (L-BFGS)"""
    G = np.size(c)
    K = int(G*(G+1)/2)
    dtheta = ut.map_theta(G)
    E = meanz_var(alpha, a, c, data)
    ### We locally define proper objective and gradient functions
    x0 = np.zeros(K)
    for k, (i,j) in dtheta.items(): x0[k] = theta[i,j]
    def f(x):
        mtheta = np.zeros((G,G))
        for k, (i,j) in dtheta.items(): mtheta[i,j] = x[k]
        return -objQ(states, alpha, data, a, mtheta, c)
    def Df(x):
        mtheta = np.zeros((G,G))
        for k, (i,j) in dtheta.items(): mtheta[i,j] = x[k]
        M = ut.meanz_brute(states, a, mtheta, c)
        Dtheta = M - E # (-1) * original gradient
        dx = np.zeros(K)
        for k, (i,j) in dtheta.items(): dx[k] = Dtheta[i,j]
        return dx
    res = scipy.optimize.minimize(f, x0, method='L-BFGS-B', jac=Df)
    if not res.success: print('Warning, Maximization step failed')
    elif info: print('M step done in {} iterations\n'.format(res.nit))
    for k, (i,j) in dtheta.items():
        theta[i,j], theta[j,i] = res.x[k], res.x[k]
    return theta

def m_step_sp(message, alpha, data, a, theta, c, info=True):
    """Perform the M step using the sum-product algorithm"""
    ngrad = cf.iter_grad
    tau = cf.learning_rate
    mix = cf.lasso_mix
    tol = cf.m_tol
    sp.count_warning = 0 # Counter of sum-product warnings
    max_val = 20 # Maximum value for theta entries
    E = meanz_var(alpha, a, c, data)
    S = ut.zscales(c) # Scaling constant for theta entries
    # states = ut.state_vector(c) # For tests
    ### Perform basic gradient method
    t = 0 # Number of iterations
    v = tol + 1 # Check variation of theta
    v0 = v
    while ((t < ngrad) & (v > tol)):
        t += 1
        ### Compute the mean matrix: this is the costly step
        M = sp.meanz(message, a, theta, c)
        ### Perform gradient step
        theta0 = theta
        Dtheta = E - M # Gradient of the objective function
        rate = tau * S # Normalized learning rate
        pen = cf.penalization * S
        theta = ut.prox_operator(theta + rate*Dtheta, rate, pen, mix)
        ### Check 1: convergence
        v = np.max(np.abs(theta - theta0))
        # print(v)
        # print(Dtheta)
        if not v < v0:
            tau *= 0.5
            # print('learning rate -> {:.2e}'.format(tau))
            if tau < tol:
                print('Warning: learning rate < {:.2e}'.format(tol))
        v0 = v
        ### Check 2: stability
        if np.max(theta) > max_val:
            theta = theta0
            t -= 1
            tau *= 0.1
            cf.learning_rate = tau
            msg = 'Warning: the gradient step is unstable: '
            msg += 'learning rate -> {:.2e}'.format(tau)
            print(msg)
    # M2 = ut.meanz_brute(states, a, theta, c) # For tests
    # print('Max diff = {}'.format(np.max(np.abs(M-M2))))
    # print(M-M2)
    cw = sp.count_warning
    if cw > 0: print('Warning: bad convergence of SP ({} times)'.format(cw))
    if info: print('M step done in {} iterations\n'.format(t))
    return theta