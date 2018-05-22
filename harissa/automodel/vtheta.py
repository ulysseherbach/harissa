"""Useful classes for the automodel package"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Vtheta:
    """
    Handle the symetric interaction matrix in a compact form,
    supporting trajectories of successive values.
    """
    def __init__(self, theta):
        G = np.size(theta[0])
        self.size = G
        self.nval = 1
        ### Create the data structure
        fields, vtheta = [], []
        for i in range(G):
            for j in range(i,G):
                fields += [('{}-{}'.format(i+1,j+1),'float64')]
                vtheta += [theta[i,j]]
        self.traj = np.array([tuple(vtheta)], dtype=fields)

    def __repr__(self):
        msg = 'Trajectory of theta'
        msg += ' ({} value{})'.format(self.nval, (self.nval > 1)*'s')
        return msg

    def append(self, theta):
        """Append a new value to a theta trajectory."""
        vtheta = Vtheta(theta)
        self.traj = np.append(self.traj, vtheta.traj)
        self.nval += 1

    def get_theta(self, t=None):
        """Build a theta matrix from one value of the trajectory,
        using the last value by default, or the value of index t."""
        G = self.size
        if t is None: vtheta = self.traj[self.nval - 1]
        else: vtheta = self.traj[t]
        mtheta, count = np.zeros((G,G)), 0
        for i in range(G):
            for j in range(i,G):
                mtheta[i,j] = vtheta[count]
                if (i != j): mtheta[j,i] = mtheta[i,j]
                count += 1
        return mtheta

    def savetxt(self, name):
        """Save a theta trajectory in a compact text form."""
        np.savetxt(name, self.traj, fmt='%.5f')

    def save(self, fname):
        """Save a theta trajectory in a compact binary form."""
        np.save(fname, self.traj)

    def plot(self, fname, *inters):
        G = self.size
        x = np.array(range(self.nval))
        if len(inters) == 0:
            inters = self.traj.dtype.names
        count = 0
        ### Prepare the figure
        fig = plt.figure(figsize=(10,6), dpi=100)
        gs = gridspec.GridSpec(2, 1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        for i in range(G):
            for j in range(i,G):
                inter = '{}-{}'.format(i+1,j+1)
                if inter in inters:
                    y = self.traj[inter]
                    lb = r'$\mathregular{\theta_{'
                    lb += '{},{}'.format(i,j) + r'}}$'
                    if (i != j):
                        if np.abs(y[-1]) < 0.1: lb = None
                        else: count += 1
                        ax2.plot(x, y, lw=1.5, label=lb)
                    else: ax1.plot(x, y, lw=1.5, ls='--', label=lb)
        if G <= 10: ax1.legend(loc='upper left')
        if count > 0 and count <= 10: ax2.legend(loc='upper left')
        fig.savefig(fname, dpi=100, bbox_inches='tight', frameon=False)
        plt.close()


### Functions
def load_vtheta(fname, size):
    """Load a theta trajectory from a formatted file."""
    vtheta = None
    ctxt = fname.endswith('.txt') or fname.endswith('.gz')
    cnpy = fname.endswith('.npy')
    if ctxt: vtheta_array = np.loadtxt(fname)
    elif cnpy: vtheta_array = np.load(fname)
    if ctxt or cnpy:
        if (len(np.shape(vtheta_array)) > 1):
            G, K = size, np.size(vtheta_array[:,0])
        else: G, K = size, 0
        types, traj = [], []
        for i in range(G):
            for j in range(i,G):
                types += [('{}-{}'.format(i+1,j+1),'float')]
        for k in range(K):
                traj += [tuple(vtheta_array[k])]
        if K == 0:
            traj, K = [tuple(vtheta_array)], 1
        traj = np.array(traj, dtype=types)
        vtheta = Vtheta(np.zeros((G,G)))
        vtheta.traj = traj
        vtheta.nval = K
    return vtheta