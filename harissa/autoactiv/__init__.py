""" Autoactiv package - detection of positive loops in gene networks from single-cell data """

import numpy as np

import autoactiv.utils as utils
import autoactiv.core as core

### Import special functions
from numpy import exp, log
from scipy.special import gamma

__version__ = '0.1'
__author__ = 'Ulysse Herbach'

############################
### The class to handle data
class scdata:
    """
    A class to handle single-cell expression data :
    1. ensure the data has a unified form (structured numpy array)
    2. add useful methods to manipulate it
    """

    def __init__(self, data):
        """Basically store the data in the proper structured numpy array"""
        self.array = data
        
    def getGenes(self, *lgenes):
        """Get the genes in the data in the form of a dictionary {idgene: name}
        Optionally select a subset of genes by id or name"""
        names = list(self.array.dtype.names)
        names.remove('idcell')
        names.remove('timepoint')
        ids = list(range(1,len(names)+1))
        genedict = dict(zip(ids,names))

        ### If specific genes are mentionned, we build the proper list
        if lgenes:
            idset = set(ids)
            nameset = set(names)
            genesubdict = dict()
            ### Build the inversed dictionary to find the id of a gene
            invgenedict = dict(zip(names,ids))
            for g in lgenes:
                if g in idset:
                    genesubdict.update({g: genedict[g]})
                elif g in nameset:
                    genesubdict.update({invgenedict[g]: g})
            genedict = genesubdict
            
        return genedict

    def getTimepoints(self):
        """Get the list of time-points in the data"""
        l = list(set(self.array['timepoint']))
        l.sort()
        return l

    def getValidData(self,generef):
        """Get all the valid data for a given gene (by id or name)
        The output is a (N,2) array where N is the number of valid measures
        NB: it is a copy of the original numpy array"""
        genelist = list(self.getGenes(generef).items())
        if (len(genelist) == 1):
            idgene, gene = genelist[0]
            ### Return the relevant view of the structured array
            X = self.array
            X = X[X[gene] >= 0] # Remove the UDs
            ### Create a copy in a more traditional (N,2) array
            Y = np.zeros((np.size(X),2))
            Y[:,0] = X['timepoint']
            Y[:,1] = X[gene]
            return Y
        else: print("Warning, no such gene found in data")

    def spreadZeros(self):
        """Replace the zeros with some consistent positive values
        Data is assumed to be in the proper imported format. For each gene:
        1. Infer the parameters of distribution gamma(a,b) (single-gene model)
        2. Replace zeros with draws of gamma(a,b+1) (posterior of the gamma-Poisson model),
        conditionned on being smaller than the minimum measured positive value of the gene."""
        genedict = self.getGenes()
        z = 0
        for idgene, gene in genedict.items():
            X = self.array[gene] # X is a view so Data is going to be modified
            xmin = np.min(X[X>0])
            ### Estimate the parameters after removing the UDs
            (a,b) = utils.estimGamma(X[X>=0])
            ### Spreading the zeros
            for k in range(0,np.size(X)):
                if (X[k] == 0):
                    xtest = xmin + 1
                    while (xtest > xmin):
                        xtest = np.random.gamma(a,1/(b+1),1)
                    X[k] = xtest
                    z += 1

            ### Monitoring the result
            spread = True
            if (np.sum(X == 0) != 0):
                spread = False
                print("Warning, still problems of zero values for gene {i} ({g})!".format(i = idgene, g = gene))

        ### Success
        if (spread and z): print("Successfully spread all zeros ({}).".format(z))
        elif spread: print("This data has only positive measurments.")

    ### How to print scdata objects
    def __repr__(self):
        T = len(list(set(self.array['timepoint'])))
        C = np.size(self.array)
        genedict = self.getGenes()
        G = len(genedict)
        Z, U = 0, 0
        for idgene, gene in genedict.items():
            X = self.array[gene]
            Z += np.size(X[X==0])
            U += np.size(X[X==-1])
        message = 60*"-"+"\n"
        message += "Single-cell dataset: {} cells ({} time-points), {} genes.\n".format(C,T,G)
        message += "Missing values: {} UDs ({:.2f} percent of the dataset).\n".format(U,100*U/(C*G))
        if Z: message += "The dataset contains {} zero values.\n".format(Z)
        else: message += "The dataset contains no zero value.\n"
        message += 60*"-"
        return message

################################################
### The class to handle the autoactivation model
class model:
    """Parameters of the autoactivation model for a single-gene"""

    def __init__(self, a, theta, c):
        """Store the parameters of the model"""
        self.a = a # a = (a[0],a[1],a[2]) where a[0] = k0, a[1] = m, a[2] = koff/s
        self.theta = theta # Dictionary {timepoint t: theta[t]}
        self.c = c # Cluster number (model = mixture of c+1 gamma distributions)

    def getTimepoints(self):
        """Get the list of time-points in the model"""
        l = list(self.theta)
        l.sort()
        return l

    def getDistribution(self,x,timepoint):
        """Get the distribution associated with the model at a given timepoint:
        - the input is a 1D numpy array of size N representing molecule numbers
        - the distribution is then computed for each value of x
        The output is a (N,T) numpy array where T is the number of time-points."""
        a, theta, c = self.a, self.theta, self.c
        B = utils.binom(c)
        z = np.linspace(0,c,c+1)
        lz = a[0] + a[1]*z
        Z = np.sum(B*exp(theta[timepoint]*z)*gamma(lz)/(a[2]**lz))
        l = (a[0]-1)*log(x) - a[2]*x + c*log(1 + exp(theta[timepoint])*(x**a[1])) - log(Z)

        return exp(l)

    ### How to print model objects
    def __repr__(self):
        timepoints = self.getTimepoints()
        ltheta = [self.theta[t] for t in timepoints]
        a, c = self.a, self.c
        message = 60*"-"+"\n"
        message += "Autoactivation model ({} time-points):\n".format(len(self.theta))
        message += "k0 = {}, k1 = {}, koff/s = {}, m = {}, c = {}\n".format(a[0],a[0] + c*a[1],a[2],a[1],c)
        message += "Timepoints: {}\n".format(timepoints)
        message += "theta: {}\n".format(ltheta)
        message += 60*"-"
        return message


################################################
### Main tools provided by the package Autoactiv
def load(data_path):
    """Load a dataset into a scdata object.
    The input is a path for a .npy file with the proper structure"""
    return scdata(np.load(data_path))

def posterior(dataset,*lgenes,**kwargs):
    """Compute the posterior for the auto-activation power
    The input is a scdata object and optionally a list of genes"""
    genedict = dataset.getGenes(*lgenes)
    Timepoints = dataset.getTimepoints()
    T = len(Timepoints) # Number of time-points
    datatypes = [('idgene',"int64"),('a',"float64",3),('theta',"float64",T),('c',"int64"),('p',"float64")]
    results_path = kwargs.get('results_path', None)
    append = kwargs.get('append', False)

    # Vc = np.array(range(10,0,-1)) # Test c values decreasing from 10 to 1
    Vc = np.array(range(1,11))
    # Vc = np.array([2])
    C = np.size(Vc)
    restot = []

    for idgene, gene in genedict.items():
        print("Inference for gene {} ({})...".format(idgene,gene))
        kwargs['idgene'] = idgene
        kwargs['gene'] = gene

        ### Initialization for a given gene
        X = dataset.getValidData(idgene)
        (k1,b) = utils.estimGamma(X[:,1])
        if (k1 < 0.01): print("Warning: data is almost 0")

        Vl = np.zeros(C) # Log-likelihood vector
        Va = np.zeros((C,3))
        Vtheta = np.zeros((C,T))

        a = np.array([k1/10,k1/Vc[0],b])
        theta = np.zeros(T)

        for i, c in enumerate(Vc):
            print("c = {}".format(c))
            ### Inference
            (a,theta) = core.inferParamEM(X,Timepoints,a,theta,c,**kwargs)
            Va[i,:] = a
            Vtheta[i,:] = theta
            Vl[i] = core.logLikelihood(X,Timepoints,a,theta,c)

        Vp = np.exp(Vl - np.min(Vl))

        ### Option: put a slight prior on c to keep small values
        mu = 0.0001 # Try 0.001
        Vp = exp(-mu*Vc)*Vp
        Vp = Vp/np.sum(Vp) # We normalize to obtain the "posterior" of c

        results = [(idgene,Va[i,:],Vtheta[i,:],c,Vp[i]) for i, c in enumerate(Vc)]
        restot += results

        if results_path:
            fname = results_path+"/Posterior.npy"
            if append:
                p = np.load(fname)
                if (np.size(p[p['idgene'] == idgene]) == 0):
                    np.save(fname, np.append(p,np.array(results, dtype = datatypes)))
                else:
                    p[p['idgene'] == idgene] = np.array(results, dtype = datatypes)
                    np.save(fname, p)
            else: np.save(fname, np.array(restot, dtype = datatypes))

    return np.array(restot, dtype = datatypes)

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




