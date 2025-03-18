"""
Sinkhorn's algorithm for resolution of static Schrödinger bridge.

References
----------
* https://personal.math.ubc.ca/~geoff/courses/W2019T1/Lecture13.pdf
* https://www.math.columbia.edu/~mnutz/docs/EOT_lecture_notes.pdf
* https://lucyliu-ucsb.github.io/posts/Sinkhorn-algorithm/
"""
import numpy as np


def relative_entropy(p, q, smooth=False):
    """
    Relative entropy of p with respect to q.

    Also known as Kullback-Leibler divergence of p from q.
    This quantity is defined only if p << q (i.e. p is absolutely
    continuous with respect to q).
    """
    if smooth and (np.any(p == 0) or np.any(q == 0)):
        p += 1e-100
        q += 1e-100
        p /= np.sum(p)
        q /= np.sum(q)
    v = np.zeros(p.shape)
    mask1 = (p > 0) * (q == 0)
    mask2 = (p > 0) * (q > 0)
    v[mask2] = p[mask2] * np.log(p[mask2]/q[mask2])
    v[mask1] = np.inf
    return np.sum(v)


def sinkhorn_bridge(k, mu, nu, tol=1e-10, verb=False):
    """
    Sinkhorn's algorithm for resolution of static Schrödinger bridge.

    NB: The algorithm solves the entropically regularized transport
    problem, where `k` is the corresponding kernel
    (i.e., k = exp(-c/epsilon)). This function provides the resulting
    coupling `p` but also the reference coupling `r` related to the
    equivalent static Schrödinger bridge problem: p = argmin{H(p|r)}
    where H(p|r) is the relative entropy of p with respect to r.
    The formula for the reference coupling (before probability
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
        print(f'{c_iter} iterations')
    # Reference coupling
    r = np.diag(mu.flatten()) @ k @ np.diag(nu.flatten())
    r /= r.sum()
    # Output
    return p, r


def entropic_coupling(mu, nu, p_ref, tol=1e-10, verb=False):
    """
    Solution of static Schrödinger problem using Sinkhorn's algorithm.

    This function solves the static Schrödinger bridge problem:
        p = argmin{H(p|p_ref)}
    for p among couplings of `mu` and `nu` such that p << p_ref, where
    H(p|p_ref) is the relative entropy of p with respect to p_ref.

    NB: This problem may have no solution in cases where `p_ref`
    contains zeros that are incompatible with couplings of `mu` and `nu`
    (however in this case, see https://arxiv.org/abs/2207.02977).
    """
    # Reshape for broadcasting
    n1, n2 = p_ref.shape
    mu = mu.reshape((n1, 1))
    nu = nu.reshape((1, n2))
    # Initialization
    p = p_ref
    p_norm = np.sum(p**2)
    u = np.ones((n1, 1))
    v = np.ones((1, n2))
    # Main loop
    delta = 1
    c_iter = 0
    while delta > tol:
        p_norm_old = p_norm
        u = mu / (p_ref @ v.T)
        v = nu / (u.T @ p_ref)
        p = p_ref * u * v
        p_norm = np.sum(p**2)
        delta = np.abs(p_norm - p_norm_old) / p_norm_old
        c_iter += 1
    if verb:
        print(f'{c_iter} iterations')
    return p


def total_variation_coupling(mu, nu):
    """
    Total variation coupling between mu and nu.

    Namely, this coupling is minimizing the quantity P(X ≠ Y) where
    X ~ mu and Y ~ nu, which is then equal to the total variation
    distance between mu and nu (i.e., 0.5 * ||mu - nu||_1).

    NB: This coupling requires that `mu` and `nu` have the same shape.
    """
    if (np.ndim(mu) != 1) or (np.ndim(nu) != 1) or (mu.size != nu.size):
        msg = 'arrays should be 1-dimensional with same size.'
        raise ValueError(msg)
    f = np.array([mu, nu])
    f_min = np.min(f, axis=0)
    p = np.sum(f_min)
    f_id = f_min / p
    f_mu = ((mu - f_min) / (1 - p)).reshape((mu.size, 1))
    f_nu = ((nu - f_min) / (1 - p)).reshape((1, mu.size))
    return p * np.diag(f_id) + (1-p) * f_mu * f_nu


# Tests
if __name__ == '__main__':

    ####################
    # Relative entropy #
    ####################
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

    #############################
    # Static Schrödinger bridge #
    #############################

    n1, n2 = 4, 5
    mu = np.ones(n1)
    nu = np.ones(n2)
    mu[1] = 1e-2
    nu[2] = 1e2
    mu /= np.sum(mu)
    nu /= np.sum(nu)

    # Define transport cost
    c = np.ones((n1, n2))
    c[1, 2] = 0
    c[2, 3] = 0

    # Define entropic transport kernel
    epsilon = 0.2
    k = np.exp(-c/epsilon)

    # Reshape for broadcasting
    p_mu = mu.reshape((n1, 1))
    p_nu = nu.reshape((1, n2))

    # Reference coupling for Schrödinger bridge (use broadcasting)
    p_ref = k * p_mu * p_nu
    p_ref /= np.sum(p_ref)

    # Option 1: define from transport kernel
    p1, r = sinkhorn_bridge(k, mu, nu, verb=True)

    # Option 2: define from reference coupling
    p2 = entropic_coupling(mu, nu, k, verb=True)

    # Output should coincide
    print(f'Coincide: {np.allclose(p1, p2) and np.allclose(r, p_ref)}')

    # Check marginals
    print(f'Marginal 1: {np.allclose(p2.sum(axis=1), mu)}')
    print(f'Marginal 2: {np.allclose(p2.sum(axis=0), nu)}')
