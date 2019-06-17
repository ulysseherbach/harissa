"""
Vector Auto-Regressive model on kinetic parameters: this can be seen as a
mechanistic analog of the SINCERITIES algorithm [Papili Gao et al., 2018].

Reference
---------
Papili Gao et al., SINCERITIES: Inferring gene regulatory networks
from time-stamped single cell transcriptional expression profiles.
Bioinformatics, 34(2):258–266, 2018.
"""
import numpy as np
from scipy import sparse

def variation_matrix(a_time, t):
        """
        Compute variations of kinetic 'a' parameters between timepoints.
        Returns an array with shape (T-1,G) where G is the number of
        genes + 1 (stimulus) and T is the number of timepoints.
        NB: The first column 'gene 0' represents the stimulus at t=0.
        """
        T, G = a_time.shape
        dt = t[1:] - t[0:T-1] # Time intervals
        scale = np.min(dt)/dt # Scales of time intervals
        # Initialize the matrix
        v = np.zeros((T-1,G))
        # Fill the matrix
        for i in range(T-1):
            v[i] = scale[i] * (a_time[i+1] - a_time[i])
        return v

def score_matrix_var(v, alpha=None, verb=False):
    """
    Compute a sparse matrix f by keeping only the most likely links,
    with f[i,j] denoting the statistical strength of influence i -> j.
    """
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import LeavePOut
    T, G = v.shape
    x = np.zeros((T-1,G))
    y = np.zeros((T-1,G))
    for t in range(T-1):
        x[t] = v[t]
        y[t] = v[t+1]
    x = sparse.csr_matrix(x)
    model = ElasticNet(fit_intercept=False, l1_ratio=0.5, max_iter=100000)
    # Possible choice of alpha by 1-fold cross validation
    if alpha is None:
        # Number of alpha values to try
        n_alphas = 10
        # Interval for alpha values
        valpha = np.logspace(-3,-1,n_alphas)
        # Vector of prediction errors
        verror = np.zeros(n_alphas)
        vtest = np.arange(T-1)
        # Cross validation
        if verb: print('Cross validation...\n[(alpha) -> (error)]:')
        for i in range(n_alphas):
            model.alpha=valpha[i]
            lpo = LeavePOut(p=1)
            for train, test in lpo.split(vtest):
                # if 0 in train:
                    model.fit(x[train], y[train])
                    verror[i] += np.sum((model.predict(x[test]) - y[test])**2)
            if verb: print('{:.2e} -> {:.5f}'.format(valpha[i], verror[i]))
        # Get the best alpha
        i = np.argmin(verror)
        if i == 0: print('Warning: alpha reaching left bound')
        elif i == n_alphas - 1: print('Warning: alpha reaching right bound')
        model.alpha = valpha[i]
    else: model.alpha = alpha
    # Final fit using the selected alpha
    if verb: print('Choosing alpha = {:.2e}'.format(model.alpha))
    model.fit(x,y)
    f = model.sparse_coef_.transpose()

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8,3), dpi=100)
    # lab = 'Prediction error'
    # plt.semilogx(valpha, verror, linewidth=2, color='red', label=lab)
    # plt.xlim(np.min(valpha), np.max(valpha))
    # plt.ylim(np.min(verror), np.max(verror))
    # plt.xlabel('alpha')
    # plt.legend()
    # file = 'cross_validation.pdf'
    # plt.savefig(file, dpi=100, bbox_inches='tight', frameon=False)
    # plt.close()

    return f
