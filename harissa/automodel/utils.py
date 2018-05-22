"""Useful functions"""
import numpy as np
from numpy import exp, log
from scipy.special import gammaln

def cov(X,Y):
    """Compute the covariance of two variables"""
    return np.mean(X*Y) - np.mean(X)*np.mean(Y)

def binom(n):
    """Binomial coefficients in the form of a vector"""
    v = np.ones(n+1)
    v[1:n+1] = np.linspace(n,1,n)/np.linspace(1,n,n)
    return np.cumprod(v)

def binomln(n):
    """Log of binomial coefficients in the form of a vector"""
    v = np.zeros(n+1)
    v[1:n+1] = log(np.linspace(n,1,n)) - log(np.linspace(1,n,n))
    return np.cumsum(v)

def estim_dist(z, c):
    """Estimate the distribution of z from a sample"""
    p = np.zeros(c+1)
    for i in range(np.size(z)):
        p[z[i]] += 1
    return p/np.sum(p)

def is_symmetric(A):
    """Check if a matrix A is square and symmetric"""
    n, p = np.shape(A)
    if (n != p): return False
    for i in range(n):
        for j in range(i+1, n):
            if (A[i,j] != A[j,i]):
                return False
    return True

def neutral_theta(a, c):
    """Compute neutral values for theta given hyperparameters."""
    theta0 = a[1] * np.log(a[2])
    theta0 += (gammaln(a[0]) - gammaln(a[0] + c*a[1]))/c
    return np.diag(theta0)

def map_theta(G):
    """Build the dictionary {k: (i,j)} for k = 0, ..., G*(G+1)/2 - 1
    corresponding to the relevant entries of matrix theta
    ordered by lexicographic order"""
    l = {}
    k = 0
    for i in range(G):
        for j in range(i,G):
            l[k] = (i,j)
            k += 1
    return l

def zscales(c):
    """Compute the "scaling matrix" S of z, i.e.,
    S[i,i] = 1/c[i] and S[i,j] = 1/(c[i]*c[j]).
    This is useful to adapt the gradient of theta."""
    G = np.size(c)
    S = np.zeros((G,G))
    for i in range(G):
        S[i,i] = 1/c[i]
        for j in range(i+1,G):
            S[i,j] = 1/(c[i]*c[j])
            S[j,i] = S[i,j]
    return S

# def prox_operator(x, rate, pen, mix):
#     """Proximal operator for the 'elastic net' penalization.
#     Only the non-diagonal elements of x are penalized."""
#     gam, l, a = rate, pen, mix
#     s = gam * l * a
#     n = np.size(x[0,:])
#     v = np.zeros((n,n))
#     if (l == 0): return x
#     elif (l > 0):
#         for i in range(n):
#             v[i,i] =  x[i,i]
#             for j in range(i+1,n):
#                 if (x[i,j] >= s):
#                     v[i,j] =  (x[i,j] - s)/(1 + gam*l*(1-a))
#                     v[j,i] =  v[i,j]
#                 elif (x[i,j] <= -s):
#                     v[i,j] =  (x[i,j] + s)/(1 + gam*l*(1-a))
#                     v[j,i] =  v[i,j]
#         return v

def prox_operator(x, rate, pen, mix):
    """
    Proximal operator for the 'elastic net' penalization.
    Only the non-diagonal elements of x are penalized.
    NB: also works if rate, pen, mix are matrices.
    """
    gam, l, a = rate, pen, mix
    s = gam * l * a
    t = np.abs(x) > s
    v = t * (x - np.sign(x)*s*t) / (1 + gam*l*(1-a))
    return v + np.diag(np.diag(x) - np.diag(v))

def store_em(theta, iter_em, init=False, vtheta=None, t=0, fname=None):
    """Store successive values of theta"""
    G = np.size(theta[0,:])
    n = int(G*(G+1)/2)
    if init:
        types = []
        theta0 = []
        for i in range(G):
            for j in range(i,G):
                types += [('{}-{}'.format(i+1,j+1),'float64')]
                theta0 += [theta[i,j]]
        theta_list = [tuple(theta0)]
        theta_list += [tuple(np.zeros(n)) for k in range(iter_em)]
        return np.array(theta_list, dtype=types)
    else:
        for i in range(G):
            for j in range(i,G):
                inter = '{}-{}'.format(i+1,j+1)
                vtheta[inter][t+1] = theta[i,j]
    if fname: np.savetxt(fname+'.txt.gz', vtheta, fmt='%.2f')

def state_vector(c, fixed=None):
    """Build the full state vector of z in lexicographic order,
    where states of z[i] are 0,...,c[i]. If fixed = {i: [z0[i]]}
    is provided, the available z[i] are clamped to z0[i]."""
    if fixed is None: fixed = {}
    n = len(c) # Number of genes
    val = {i:fixed.get(i, range(c[i]+1)) for i in range(n)}
    states = [(k,) for k in val[n-1]]
    for i in range(n-2,-1,-1):
        l = []
        for k in val[i]:
            l = l + [(k,) + u for u in states]
        states = l
    return states

def statistics(x):
    """Sufficient statistics for one cell (including missing data)"""
    G = np.size(x)
    D = np.zeros((3,G))
    for g in range(G):
        if (x[g] > 0):
            D[0,g] = 1
            D[1,g] = log(x[g])
            D[2,g] = -x[g]
    return D

def log_likelihood_estim(data, z, a):
    """Estimate the normalized log-likelihood by Monte Carlo"""
    C, G = np.shape(data)
    K = np.size(z[:,0]) # Sample size
    B = a[2]
    L = 0 # Normalized log-likelihood
    for cell in range(C):
        X = data[cell]
        D = statistics(X)
        Q = 0 # Approximate likelihood of each cell
        for k in range(K):
            Kon = a[0] + a[1]*z[k]
            S = np.sum((Kon-1)*D[1]+B*D[2]+(Kon*log(B)-gammaln(Kon))*D[0])
            Q += exp(S)
        Q = Q/K
        if (Q > 0):
            L += log(Q)
    return L/C

def cst_brute(states, a, theta, c):
    """Compute the normalizing constant by full sum"""
    A = 0
    G = np.size(c)
    B = [binom(v) for v in c] # List of needed binomial coefficients
    for z in states:
        b = np.array([B[i][z[i]] for i in range(G)])
        l = a[0] + a[1]*z
        R = np.zeros((G,G))
        for i in range(G):
            R[i,i] = theta[i,i]*z[i]
            for j in range(i+1,G):
                R[i,j] = theta[i,j]*z[i]*z[j]
        p = exp(np.sum(R) + np.sum(gammaln(l) - l*log(a[2]))) * np.prod(b)
        A += p
    return A

def meanz_brute(states, a, theta, c):
    """Compute the mean matrix for z by full sum (small networks)"""
    A = 0 # Normalizing constant
    G = np.size(c)
    M = np.zeros((G,G))
    B = [binom(v) for v in c] # List of needed binomial coefficients
    for z in states:
        b = np.array([B[i][z[i]] for i in range(G)])
        l = a[0] + a[1]*z
        R = np.zeros((G,G))
        for i in range(G):
            R[i,i] = theta[i,i]*z[i]
            for j in range(i+1,G):
                R[i,j] = theta[i,j]*z[i]*z[j]
        p = exp(np.sum(R) + np.sum(gammaln(l) - l*log(a[2]))) * np.prod(b)
        A += p
        for i in range(G):
            M[i,i] += z[i]*p
            for j in range(i+1,G):
                M[i,j] += z[i]*z[j]*p
    ### Option: symmetrize M
    for i in range(G):
        for j in range(i+1,G): 
                M[j,i] = M[i,j]
    return M/A

def log_likelihood_exact(data, a, theta, c):
    """Exact normalized logLikelihood (small networks)"""
    C, G = np.shape(data)
    states = state_vector(c)
    B = [binomln(v) for v in c] # Needed log-binomial coefficients
    L = 0
    ### 1. Compute normalizing constant
    A = 0
    for z in states:
        b = np.array([B[i][z[i]] for i in range(G)])
        l = a[0] + a[1]*z
        R = np.zeros((G,G))
        for i in range(G):
            R[i,i] = theta[i,i]*z[i]
            for j in range(i+1,G):
                R[i,j] = theta[i,j]*z[i]*z[j]
        A += exp(np.sum(R) + np.sum(gammaln(l) - l*log(a[2]) + b))
    ### 2. Compute mRNA density
    for cell in range(C):
        x = data[cell]
        U = 0 # Density
        for z in states:
            b = np.array([B[i][z[i]] for i in range(G)])
            l = a[0] + a[1]*z
            R = np.zeros((G,G))
            v = np.zeros(G)
            for i in range(G):
                R[i,i] = theta[i,i]*z[i]
                for j in range(i+1,G):
                    R[i,j] = theta[i,j]*z[i]*z[j]
                ### Check for missing data
                if x[i] > 0: v[i] = (l[i]-1)*log(x[i]) - a[2,i]*x[i]
                else: v[i] = gammaln(l[i]) - l[i]*log(a[2,i])
            U += exp(np.sum(R) + np.sum(v + b))
        L += log(U)
    L = L/C - log(A)
    return L

def entropy_mc(z):
    """Estimate Shannon entropy of Z using a sample.
    Theoretically this is the only estimator that converges
    to the exact value for a large enough sample. By contrast,
    entropy_varmc would need a infinite number of cells."""
    K, G = np.shape(z)
    statecount, p = {}, []
    for k in range(K):
        s = tuple(z[k])
        if statecount.get(s): statecount[s] += 1
        else: statecount[s] = 1
    for count in statecount.values(): p.append(count)
    p = np.array(p)
    p = p/np.sum(p)
    return -np.sum(p*np.log2(p))

def entropy_varmc(z, alpha, c):
    """Estimate Shannon entropy of Z using both a sample
    and the variational parameters related to observations."""
    K, G = np.shape(z)
    C = np.size(alpha[:,0])
    S, B = 0, [binom(v) for v in c]
    for k in range(K):
        p, Z = 0, z[k]
        b = np.array([B[i][Z[i]] for i in range(G)])
        for cell in range(C):
            A = alpha[cell]
            v = log(b) + A*Z - c*log(1 + exp(A))
            p += exp(np.sum(v))
        p = p/C
        if (p > 0): S += np.log2(p)
        else: K -= 1
    return -S/K

def entropy_varbrute(alpha, c):
    """Estimate Shannon entropy of Z using only the variational parameters.
    It is likely to be the more precise option but it requires
    performing the brutal sum over all Z states."""
    C, G = np.shape(alpha)
    ### First we estimate the marginal distribution of z using alpha
    states = state_vector(c)
    B = [binom(v) for v in c] # List of needed binomial coefficients
    E = 0
    for z in states:
        b = np.array([B[i][z[i]] for i in range(G)])
        p = 0 # Estimation of p(z)
        for cell in range(C):
            a = alpha[cell]
            p += np.prod(b*exp(a*z)/((1 + exp(a))**c))
        p = p/C
        E -= p*np.log2(p)
    return E