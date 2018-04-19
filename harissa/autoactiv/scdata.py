"""The class to handle data"""
import numpy as np
from .utils import estimGamma

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
            (a,b) = estimGamma(X[X>=0])
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
        if (spread and z): print("Successfully spread all zeros ({}).".format(z))
        elif spread: print("This data has only positive measurments.")

    def __repr__(self):
        """How to print scdata objects"""
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