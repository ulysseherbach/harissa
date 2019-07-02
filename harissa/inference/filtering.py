"""
Focus on relevant genes by filtering
"""
import numpy as np
from scipy import sparse
from .sincerities import score_matrix
from .var import score_matrix_var

# OPTION 1: SINCERITIES method
def distance_best(d, threshold, verb=False):
    """
    Remove the less variable genes
    """
    T, G = d.shape
    s = threshold
    genes = np.arange(G)
    dmax = np.max(d, axis=0)
    select = dmax > s
    d = d[:,select]
    genes = genes[select]
    if verb:
        n = genes.size - 1
        print('Selected genes (threshold = {}): '
            '{} out of {} ({:.2%})'.format(s, n, G-1, n/(G-1)))
    return d, genes

def network_filter(d, threshold, alpha=None, l1=0.5, verb=False):
    """
    Compute a sparse matrix f by keeping only the most likely links,
    with f[i,j] = 1 denoting the possibility of interaction i -> j.
    """
    T, G = d.shape
    dnew, genes = distance_best(d, threshold, verb=verb)
    fnew = score_matrix(dnew, alpha=alpha, l1=l1, verb=verb)
    f = sparse.dok_matrix((G,G))
    # f = sparse.dok_matrix((G,G), dtype='uint')
    I, J = fnew.nonzero()
    for i, j in zip(I,J):
        f[genes[i],genes[j]] = fnew[i,j]
        # f[genes[i],genes[j]] = 1
    if verb:
        n = fnew.count_nonzero()
        n0 = G*(G-1)
        print('Number of possible interactions: {} '
            'out of {} ({:.5%})'.format(n, n0, n/n0))
    return f.asformat('csc')

# OPTION 2: mechanistic method
def genes_best(data, threshold, verb=False):
    """
    Get a smaller list of genes by removing the less variable ones.
    """
    G = data[0].size
    s = threshold
    genes = np.arange(1,G)
    x = data[:,1:]

    # # OPTION 1: threshold using the mean level
    # level = np.mean(x, axis=0)

    # # OPTION 2: threshold using the maximum level
    # level = np.max(x, axis=0)

    # OPTION 3: threshold using burst frequencies
    times = data[:,0]
    t = np.sort(list(set(times)))
    T = t.size
    a = np.zeros((T,G-1))
    # Rough estimation of 'a' for each time point
    for i in range(T):
        cells = (times == t[i])
        m = np.mean(x[cells], axis=0)
        v = np.var(x[cells], axis=0)
        e = 1e-3 * np.min(v[v > 0])
        a[i] = (v > 0) * m*m/(v+e)
    # # OPTION 3a: standard deviation
    # level = np.sqrt(np.var(a, axis=0))
    # OPTION 3b: total range
    level = np.max(a, axis=0) - np.min(a, axis=0)

    select = level >= s
    genes = genes[select]
    if verb:
        n = genes.size
        print('Selected genes (threshold = {}): '
            '{} out of {} ({:.2%})'.format(s, n, G-1, n/(G-1)))
    return list(genes)

def network_filter_mechanistic(n_genes, v, genes, alpha=None, verb=False):
    """
    Compute a sparse matrix f by keeping only the most likely links,
    with f[i,j] = 1 denoting the signed weight of interaction i -> j.
    """
    G = n_genes
    genes = [0] + genes
    f0 = score_matrix_var(v, alpha=alpha, verb=verb)
    f = sparse.dok_matrix((G,G))
    I, J = f0.nonzero()
    for i, j in zip(I,J):
        f[genes[i],genes[j]] = f0[i,j]
    if verb:
        n = f0.count_nonzero()
        n0 = G*(G-1)
        print('Number of possible interactions: {} '
            'out of {} ({:.5%})'.format(n, n0, n/n0))
    return f.asformat('csc')
