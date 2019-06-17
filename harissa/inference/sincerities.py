"""
This module implements the SINCERITIES algorithm [Papili Gao et al., 2018].

Reference
---------
Papili Gao et al., SINCERITIES: Inferring gene regulatory networks
from time-stamped single cell transcriptional expression profiles.
Bioinformatics, 34(2):258–266, 2018.
"""
import numpy as np
from scipy import sparse

def ksdistance(x1, x2):
    """
    Compute the Kolmogorov-Smirnov distance between
    empirical distributions of two datasets x1 and x2.
    """
    x = np.sort(list(set(x1).union(set(x2))))
    n, n1, n2 = x.size, x1.size, x2.size
    x1 = np.append(np.sort(x1), x[n-1]+1)
    x2 = np.append(np.sort(x2), x[n-1]+1)
    # Cumulative distribution functions
    y1, y2 = np.zeros(n), np.zeros(n)
    k1, k2 = 0, 0
    for k in range(n):
        while x1[k1] <= x[k]: k1 += 1
        while x2[k2] <= x[k]: k2 += 1
        y1[k] = k1/n1
        y2[k] = k2/n2
    return np.max(np.abs(y1 - y2))

def distance_matrix(data, verb=False):
        """
        Compute all Kolmogorov-Smirnov distances between timepoints.
        Returns an array with shape (T-1,G) where G is the number of
        genes + 1 (stimulus) and T is the number of timepoints.
        NB: The first column 'gene 0' represents the stimulus at t=0.
        """
        t = np.sort(list(set(data[:,0]))) # Time points
        G = data[0].size # Number of genes + stimulus
        T = t.size # Number of time points
        dt = t[1:] - t[0:T-1] # Time intervals
        scale = np.min(dt)/dt # Scales of time intervals
        # Initialize the matrix
        d = np.zeros((T-1,G))
        d[0,0] = scale[0] * 1 # Symbolize the stimulus
        # Fill the matrix
        for g in range(1,G):
            if verb: print('Computing distances for gene {}'.format(g))
            times = data[:,0]
            x = data[:,g]
            for i in range(T-1):
                x1 = x[times==t[i]]
                x2 = x[times==t[i+1]]
                d[i,g] = scale[i] * ksdistance(x1,x2)
        return d

def score_matrix(d, alpha=None, verb=False):
    """
    Compute a sparse matrix f by keeping only the most likely links,
    with f[i,j] denoting the statistical strength of influence i -> j.
    """
    from sklearn.linear_model import ElasticNet
    T, G = d.shape
    x = np.zeros((T-1,G))
    y = np.zeros((T-1,G))
    for t in range(T-1):
        x[t] = d[t]
        y[t] = d[t+1]
    x = sparse.csr_matrix(x)
    model = ElasticNet(fit_intercept=False, positive=True)
    # Possible choice by 1-fold cross validation
    if alpha is None:
        n_alphas = 10
        valpha = np.logspace(-3,-2,n_alphas)
        # n_alphas = 1
        # valpha = np.logspace(-1,-1,n_alphas)
        verror = np.zeros(n_alphas)
        vtest = np.arange(T-1)
        if verb: print('Cross validation (alpha -> error):')
        for i in range(n_alphas):
            # Cross validation
            model.alpha=valpha[i]
            for t in range(T-1):
                x_train, x_test = x[vtest!=t], x[vtest==t]
                y_train, y_test = y[vtest!=t], y[vtest==t]
                model.fit(x_train,y_train)
                verror[i] += np.sum((model.predict(x_test) - y_test)**2)
            if verb:
                print('{:.2e} -> {:.3f}'.format(valpha[i], verror[i]))
        vi = np.argmin(verror)
        if np.min(vi) == 0:
            print('Warning: the best alpha is reaching left bound')
        elif np.max(vi) == n_alphas - 1:
            print('Warning: the best alpha is reaching right bound')
        # Final fit using the best alpha
        model.alpha = np.mean(valpha[vi])
    else: model.alpha = alpha
    if verb: print('Choosing alpha = {:.2e}'.format(model.alpha))
    model.fit(x,y)
    f = model.sparse_coef_.transpose()
    return f

def sincerities(d, data, verb=False):
    """
    Compute a sparse matrix f by keeping only the most likely links,
    with f[i,j] denoting the strength of influence i -> j with its sign.
    """
    # Prepare the data
    x = data.copy()
    x[:,0] = 1 * (x[:,0] == 0)
    for j in range(x[0].size):
        x[:,j] = x[:,j]/np.var(x[:,j])
    # Estimate partial correlations
    from sklearn.covariance import EmpiricalCovariance
    model = EmpiricalCovariance()
    model.fit(x)
    p = model.precision_
    # Add signs to the score matrix
    s = score_matrix(d, verb=verb).copy()
    I, J = s.nonzero()
    for i, j in zip(I,J):
        s[i,j] = np.sign(p[i,j]) * s[i,j]
    return s
