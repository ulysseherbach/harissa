"""A class for storing and manipulating the results"""
import numpy as np
from . import config as cf

class Result:
    """Object containing the result of an inference procedure."""
    
    def __init__(self, theta, alpha, sample, method,
        traj_logl=None, traj_theta=None, em_count=0,
        ent_mc=None, ent_varmc=None, ent_varbrute=None):
        self.theta = theta
        self.alpha = alpha
        self.sample = sample
        self.method = method
        self.traj_logl = traj_logl
        self.traj_theta = traj_theta
        self.em_count = em_count
        entdict = {
            'ent_mc':ent_mc,
            'ent_obs_mc':ent_varmc,
            'ent_obs_brute':ent_varbrute
        }
        self.entropy = entdict

    def __repr__(self):
        G = np.size(self.theta[0])
        message = 'Inference result for {} genes:\n'.format(G)
        message += 'Method: {}\n'.format(self.method)
        for i in range(G):
            for j in range(i+1,G):
                w = self.theta[i,j]
                if (np.abs(w) >= 0.1):
                    message += '{} - {} ({:.1f})\n'.format(i+1,j+1,w)
        return message

    def save(self, path, name, alpha=False, traj=False, recap=False):
        G = np.size(self.theta[0])
        ### Save theta
        if recap: fname = path + 'theta.txt.gz'
        else: fname = path + 'theta{}.txt.gz'.format(self.em_count)
        np.savetxt(fname, self.theta, fmt='%.2f', delimiter='\t')
        ### Save alpha
        if alpha:
            fname = path + 'alpha.txt.gz'
            np.savetxt(fname, self.alpha, fmt='%.2f', delimiter='\t')
        ### Save the inference trajectory
        if traj:
            lt = self.traj_logl
            if lt is not None:
                fname = path + 'vlogl.txt'
                np.savetxt(fname, lt, fmt='%.2f', header='logL')
            vt = self.traj_theta
            if vt is not None:
                names = vt.dtype.names
                h = str(names[0])
                for r in range(1,len(names)):
                    h += ', ' + str(names[r])
                fname = path + 'vtheta.txt.gz'
                np.savetxt(fname, vt, fmt='%.2f', delimiter=',', header=h)
        ### Export a compact recap of the results
        if recap:
            summary = 'Inference results for data [{}] '.format(name)
            summary += '({} genes)'.format(G)
            line = '-'*len(summary)
            summary = line + '\n' + summary + '\n' + line + '\n\n'
            ### Export the EM details
            summary += 'Parameters\n' + '----------\n'
            summary += 'Method: {}\n'.format(self.method)
            iter_em = np.size(self.traj_logl) - 1
            summary += 'Last EM iterations: {}\n'.format(iter_em)
            summary += 'Total EM iterations: {}\n\n'.format(self.em_count)
            ### Export the other inference parameters
            summary += 'SAMPLE_SIZE = {}\n'.format(cf.sample_size)
            summary += 'ITER_GIBBS_INIT = {}\n'.format(cf.iter_gibbs_init)
            summary += 'ITER_GIBBS = {}\n'.format(cf.iter_gibbs)
            summary += 'VAR_TOL = {}\n'.format(cf.var_tol)
            summary += 'VAR_ITER_MAX = {}\n'.format(cf.var_iter_max)
            summary += 'PENALIZATION = {}\n'.format(cf.penalization)
            summary += 'LASSO_MIX = {}\n'.format(cf.lasso_mix)
            summary += 'LEARNING_RATE = {}\n'.format(cf.learning_rate)
            summary += 'ITER_GRAD = {}\n\n'.format(cf.iter_grad)
            ### Export entropy values
            summary += 'Results\n' + '-------\n'
            ent = self.entropy
            summary += 'Model entropy: {:.2f}\n'.format(ent['ent_mc'])
            summary += 'Data entropy: {:.2f}\n\n'.format(ent['ent_obs_mc'])
            ### Export the relevant interactions
            for i in range(G):
                for j in range(i+1,G):
                    w = self.theta[i,j]
                    if (w >= 0.1):
                        summary += '{} + {}\t({:.1f})\n'.format(i+1,j+1,w)
                    elif (w <= -0.1):
                        summary += '{} - {}\t({:.1f})\n'.format(i+1,j+1,-w)
            fname = path + 'recap.txt'
            with open(fname, mode='w') as file:
                file.write(summary)