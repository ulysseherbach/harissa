### utils.py - Useful functions ###

import numpy as np

def cov(X,Y):
    """Compute the covariance of two variables"""
    return np.mean(X*Y) - np.mean(X)*np.mean(Y)

def estimGamma(X):
    """Estimate the parameters of a gamma distribution using the method of moments
    The output is (a,b) for the distribution f(x) = x**(a-1)*exp(-b*x)/(gamma(a)/b**a)"""
    m = np.mean(X)
    v = np.var(X)
    return (m*m/v, m/v)

def binom(n):
    """Binomial coefficients in the form of a vector"""
    v = np.ones((1,n+1))
    v[0,1:n+1] = np.linspace(n,1,n)/np.linspace(1,n,n)
    return np.cumprod(v)

def estimRep(X):
    """Estimate the repartition function of a sample"""
    l = np.size(X)
    x = np.append(np.append(0,np.sort(X)),1.2*np.max(X))
    y = np.append(np.linspace(0,1,l+1),1)
    return (x,y)

def funRep(x,y):
    """Compute the repartition function of a given discretized density"""
    N = np.size(x)
    F, S = np.zeros(N), 0
    for i in range(1,N):
        S += (x[i]-x[i-1])*(y[i]+y[i-1])/2
        F[i] = S
    return F

def estimBeta(X):
    # On utilise les estimateurs naturels des moments
    e1 = np.mean(X)
    e2 = np.mean(X**2)
    e3 = np.mean(X**3)

    r1 = e1
    r2 = e2/e1
    r3 = e3/e2
    A = r1*r2 - 2*r1*r3 + r2*r3
    B = r1 - 2*r2 + r3
    # Astuce : on prend la valeur absolue
    kon = np.abs(2*r1*(r3 - r2)/A)
    koff = np.abs(2*(r2 - r1)*(r1 - r3)*(r3 - r2)/(A*B))
    s = np.abs(A/B)

    return (kon, koff/s)



