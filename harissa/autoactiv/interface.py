"""Interface between core and user functions"""
import numpy as np
from .utils import estimGamma
from .core import logLikelihood, inferParamEM
from .scdata import scdata
from .model import model

def posterior(dataset,*lgenes,**kwargs):
    """Compute the pseudo-posterior for the auto-activation power.
    The input is a scdata object and optionally a list of genes"""
    genedict = dataset.getGenes(*lgenes)
    Timepoints = dataset.getTimepoints()
    T = len(Timepoints) # Number of time-points
    datatypes = [('idgene',"int64"),('a',"float64",3),('theta',"float64",T),('c',"int64"),('p',"float64")]
    results_path = kwargs.get('results_path', None)
    append = kwargs.get('append', False)
    info = kwargs.get('info', False)
    lc = kwargs.get('lc', None)
    if lc is not None: Vc = np.array(lc)
    else: Vc = np.array(range(1,11))
    C = np.size(Vc)
    restot = []
    for idgene, gene in genedict.items():
        if info: print('Inference for gene {} ({})...'.format(idgene,gene))
        kwargs['idgene'] = idgene
        kwargs['gene'] = gene
        ### Initialization for a given gene
        X = dataset.getValidData(idgene)
        (k1,b) = estimGamma(X[:,1])
        if (k1 < 0.01): print('Warning: data is almost 0')
        Vl = np.zeros(C) # Log-likelihood vector
        Va = np.zeros((C,3))
        Vtheta = np.zeros((C,T))
        a = np.array([k1/10,k1/Vc[0],b])
        theta = np.zeros(T)
        for i, c in enumerate(Vc):
            if info: print('c = {}'.format(c))
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
            fname = results_path+"/Posterior.npy"
            if append:
                p = np.load(fname)
                if (np.size(p[p['idgene'] == idgene]) == 0):
                    np.save(fname, np.append(p,np.array(results, dtype=datatypes)))
                else:
                    p[p['idgene'] == idgene] = np.array(results, dtype=datatypes)
                    np.save(fname, p)
            else: np.save(fname, np.array(restot, dtype=datatypes))
    return np.array(restot, dtype=datatypes)

def mapestim(dataset, posterior, *lgenes):
    """Compute the best models from a given posterior parameter distribution:
    - the models correspond to the maximum a posteriori (MAP) estimator
    - the output is a dictionary of model objects {gene: model}"""
    genedict = dataset.getGenes(*lgenes)
    geneset = set(posterior['idgene']) & set(genedict)
    Timepoints = dataset.getTimepoints()
    modeldict = dict()
    for idgene in geneset:
        P = posterior[posterior['idgene'] == idgene]
        Vp = P['p']
        m = np.max(Vp)
        if not np.isnan(m):
            a, theta, c = P['a'][Vp == m][0], P['theta'][Vp == m][0], P['c'][Vp == m][0]
            if isinstance(theta,float): theta = [theta]
            modeldict[genedict[idgene]] = model(tuple(a),dict(zip(Timepoints,theta)),c)
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