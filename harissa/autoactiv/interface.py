"""Interface between core and user functions"""
import numpy as np
from .utils import estimGamma
from .core import logLikelihood, inferParamEM
from .scdata import scdata
from .model import model

def posterior(dataset,*lgenes,**kwargs):
    """Compute the pseudo-posterior for the auto-activation power.
    The input is a scdata object and optionally a list of genes"""
    genedict = dataset.genes(*lgenes)
    Timepoints = dataset.timepoints()
    T = len(Timepoints) # Number of time-points
    datatypes = [('idgene',"int64"),('a',"float64",3),
        ('theta',"float64",T),('c',"int64"),('p',"float64")]
    results_path = kwargs.get('results_path', None)
    append = kwargs.get('append', False)
    info = kwargs.get('info', False)
    lc = kwargs.get('lc', None)
    if lc is None:
        ### The default range for c
        Vc = np.array([1], dtype='int')
        # Vc = np.array(range(1,11))
    else: Vc = np.array(lc)
    C = np.size(Vc)
    ### Option: use a separation to initialize parameters
    s = kwargs.get('s', 0)
    restot = []
    for idgene, gene in genedict.items():
        if info: print('Inference for gene {} ({})...'.format(idgene,gene))
        kwargs['idgene'] = idgene
        kwargs['gene'] = gene
        ### Initialization for a given gene
        Vl = np.zeros(C) # Log-likelihood vector
        Va = np.zeros((C,3))
        Vtheta = np.zeros((C,T))
        X = dataset.get_valid_data(idgene)
        # k1, b = estimGamma(X[:,1])
        k1, b = estimGamma(X[X[:,1]>=s,1])
        # print(k1,b)
        if (k1 < 0.01): print('Warning: data is almost 0')
        ### Parameters
        a = np.array([k1/10,k1/Vc[0],b])
        theta = np.zeros(T)
        for i, c in enumerate(Vc):
            # if info: print('c = {}'.format(c))
            # ### Re-initialization
            # a = np.array([k1/10,k1/Vc[i],b])
            # theta = np.zeros(T)
            ### Inference
            (a,theta) = inferParamEM(X,Timepoints,a,theta,c,**kwargs)
            Va[i,:] = a
            Vtheta[i,:] = theta
            Vl[i] = logLikelihood(X,Timepoints,a,theta,c)
        Vp = np.exp(Vl - np.min(Vl))
        ### Option: put a slight prior on c to keep small values
        # mu = 0.0001 # Try 0.001
        # Vp = np.exp(-mu*Vc)*Vp
        Vp = Vp/np.sum(Vp) # We normalize to obtain the "posterior" of c
        ### Export the results
        results = [(idgene,Va[i,:],Vtheta[i,:],c,Vp[i]) for i, c in enumerate(Vc)]
        restot += results
        if results_path:
            fname = results_path+'/Posterior.npy'
            if append:
                p = np.load(fname)
                if (np.size(p[p['idgene'] == idgene]) == 0):
                    np.save(fname, np.append(p,np.array(results, dtype=datatypes)))
                else:
                    p[p['idgene'] == idgene] = np.array(results, dtype=datatypes)
                    np.save(fname, p)
            else: np.save(fname, np.array(restot, dtype=datatypes))
    return np.array(restot, dtype=datatypes)

def mapestim(dataset, posterior, *lgenes, **kwargs):
    """Compute the best models from a given posterior parameter distribution:
    - the models correspond to the maximum a posteriori (MAP) estimator
    - the output is a dictionary of model objects {gene: model}"""
    genedict = dataset.genes(*lgenes)
    geneset = set(posterior['idgene']) & set(genedict)
    Timepoints = dataset.timepoints()
    modeldict = dict()
    for idgene in geneset:
        P = posterior[posterior['idgene'] == idgene]
        Vp = P['p']
        m = np.max(Vp)
        if not np.isnan(m):
            a, theta, c = P['a'][Vp == m][0], P['theta'][Vp == m][0], P['c'][Vp == m][0]
            if isinstance(theta,float): theta = [theta]
            modeldict[genedict[idgene]] = model(tuple(a),dict(zip(Timepoints,theta)),c)
    ### Save the results
    fname = kwargs.get('fname', None)
    if fname is not None:
        n = len(geneset)
        T = len(Timepoints)
        x = np.zeros((n,5+T))
        for k, idgene in enumerate(geneset):
            gene = genedict[idgene]
            m = modeldict[gene]
            ltheta = [m.theta[t] for t in Timepoints]
            x[k] = np.array([idgene,m.a[0],m.a[1],m.a[2]]+ltheta+[m.c])
        header = 'id a[0] a[1] a[2] theta[0] ... theta[T] c'
        fmt = ['%u'] + 3*['%.3e'] + T*['%+.2e'] + ['%u']
        np.savetxt(fname, x, fmt=fmt, delimiter='\t', header=header)
    return modeldict

def infer(data, info=False, lc=None):
    """Inference of the auto-activation model for one gene.

    Input
    -----
    data : numpy array with 2 columns
        Each row is a cell. For each cell k:
        - data[k][0] = time-point
        - data[k][1] = mRNA level

    Output
    ------
    res : an object of class autoactiv.model
        model.a = (k0, m, koff/s)
        model.theta = {timepoint t: theta[t]}
        model.c = cluster number
        NB: hence k1 = a[0] + c*a[1] and the model
        is a mixture of c+1 gamma distributions."""
    types = [ ('idcell', 'int64'), ('timepoint', 'float64'),
        ('Gene', 'float64') ]
    N = np.size(data[:,0]) # Number of cells
    ldata = [(k,data[k,0],data[k,1]) for k in range(N)]
    array = np.array(ldata, dtype=types)
    ### Format the data
    cells = scdata(array)
    ### Perform inference
    p = posterior(cells, info=info, lc=lc)
    modeldict = mapestim(cells, p)
    res = modeldict['Gene']
    return res

def get_param(dataset, posterior, timepoints, *lgenes):
    """Get the hyperparameters a and c for subsequent network inference.
    Also return theta (averaged over timepoints) as a starting point.
    NB: to be used with scdata.get_data(timepoints, *lgenes)."""
    if np.shape(timepoints) == (): timepoints = [timepoints]
    modeldict = mapestim(dataset, posterior, *lgenes)
    genedict = dataset.genes(*lgenes)
    idgenes = list(genedict.keys())
    idgenes.sort()
    genes = [genedict[i] for i in idgenes]
    a = np.array([modeldict[gene].a for gene in genes]).T
    c = np.array([modeldict[gene].c for gene in genes])
    ltheta = []
    for gene in genes:
        dtheta = modeldict[gene].theta
        ltheta.append(np.array([dtheta[t] for t in timepoints]))
    theta = np.mean(ltheta, axis=1)
    return a, theta, c