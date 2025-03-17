"""
Sinkhorn's algorithm for resolution of static Schrödinger bridge
References:
* Introduction to Entropic Optimal Transport - Marcel Nutz - 2022/12/05
* https://personal.math.ubc.ca/~geoff/courses/W2019T1/Lecture13.pdf
* https://lucyliu-ucsb.github.io/posts/Sinkhorn-algorithm/
"""
import numpy as np

def relative_entropy(p, q, smooth=False):
    """
    Relative entropy of p with respect to q (also known as
    Kullback–Leibler divergence of p from q). This quantity is defined
    only if p << q (i.e. p is absolutely continuous with respect to q).
    """
    if smooth and (np.any(p == 0) or np.any(q == 0)):
        p = p + 1e-100
        q = q + 1e-100
        p /= np.sum(p)
        q /= np.sum(q)
    return np.sum(p * np.log(p/q))

def sinkhorn_bridge(k, mu, nu, tol=1e-10, verb=False):
    """
    Sinkhorn's algorithm for resolution of static Schrödinger bridge.
    NB: The algorithm solves the entropically regularized transport problem,
    where `k` is the corresponding kernel (i.e., k = exp(-c/epsilon)).
    This function provides the resulting coupling `p` but also the reference
    coupling `r` related to the equivalent static Schrödinger bridge problem:
    p = argmin{H(p|r)} where H(p|r) is the relative entropy of p with respect
    to r. The formula for the reference coupling (before probability
    normalization) is:
        r = np.diag(mu) @ k @ np.diag(nu)
    which in general is not proportional to k.
    """
    # Initialization
    n1, n2 = k.shape
    mu = mu.reshape((n1, 1))
    nu = nu.reshape((n2, 1))
    u = np.ones((n1, 1))
    v = np.ones((n2, 1))
    p = np.diag(u.flatten()) @ k @ np.diag(v.flatten())
    p_norm = np.sum(p**2)
    # Main loop
    delta = 1
    c_iter = 0
    while delta > tol:
        p_norm_old = p_norm
        u = mu / (k @ v)
        v = nu / (k.T @ u)
        p = np.diag(u.flatten()) @ k @ np.diag(v.flatten())
        p_norm = np.sum(p**2)
        delta = np.abs(p_norm - p_norm_old) / p_norm_old
        c_iter += 1
    if verb:
        print(f"{c_iter} iterations")
    # Reference coupling
    r = np.diag(mu.flatten()) @ k @ np.diag(nu.flatten())
    r /= r.sum()
    # Output
    return p, r


# Tests
if __name__ == '__main__':

    # Relative entropy
    n = 4
    p, q = np.ones(n), np.ones(n)
    p[1] = 1e-2
    q[2] = 1e-2
    p /= np.sum(p)
    q /= np.sum(q)
    print(p)
    print(q)
    h = relative_entropy(p, q)
    print(h)

    # Static Schrödinger bridge
    n1, n2 = 4, 5
    mu = np.ones(n1)
    nu = np.ones(n2)
    mu[1] = 1e-2
    nu[2] = 1e2
    mu /= np.sum(mu)
    nu /= np.sum(nu)
    c = np.ones((n1, n2))
    c[1,2] = 0
    c[2,3] = 0
    epsilon = 0.2
    k = np.exp(-c/epsilon)
    p, r = sinkhorn_bridge(k, mu, nu, verb=True)

    # Check marginals
    print(np.allclose(p.sum(axis=1), mu))
    print(np.allclose(p.sum(axis=0), nu))

    # Check optimality
    h_optim = relative_entropy(p, r)
    for k in range(1000):
        # Random coupling
        q = np.random.gamma(1, size=(n1, n2))
        q /= np.sum(q)
        h_test = relative_entropy(q, r)
        if h_test < h_optim:
            print(f"Not optimal: {h_test} < {h_optim}")
