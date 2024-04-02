"""
Perform simulations using the PDMP model
"""
import numpy as np
from scipy.special import expit

class BurstyPDMP:
    """
    Bursty PDMP version of the network model (promoters not described)
    """
    def __init__(self, a, d, basal, inter, thin_adapt=True):
        # Kinetic parameters
        G = basal.size
        D0, D1 = d[0], d[1]
        K0, K1, B = a[0]*d[0], a[1]*d[0], a[2]
        S1 = D0*D1*a[2]/K1 # Normalize protein scales
        types = [('D0','float'), ('D1','float'), ('S1','float'),
            ('K0','float'), ('K1','float'), ('B','float')]
        plist = [(D0[i], D1[i], S1[i], K0[i], K1[i], B[i]) for i in range(G)]
        self.param = np.array(plist, dtype=types)
        # Network parameters
        self.basal = basal
        self.inter = inter
        # Default state
        types = [('M','float'), ('P','float')]
        self.state = np.array([(0,0) for i in range(G)], dtype=types)
        # Thinning parameter
        self.thin_cst = None if thin_adapt else np.sum(K1[1:])

    def kon(self, p):
        """
        Interaction function kon (off->on rate), given protein levels p.
        """
        K0, K1 = self.param['K0'], self.param['K1']
        sigma = expit(self.basal + p @ self.inter)
        Kon = (1-sigma)*K0 + sigma*K1
        Kon[0] = 0 # Ignore stimulus
        return Kon

    def kon_bound(self):
        """
        Compute the current kon upper bound.
        """
        M, P = self.state['M'], self.state['P']
        D0, D1, S1 = self.param['D0'], self.param['D1'], self.param['S1']
        # Explicit upper bound for P
        time = np.log(D0/D1)/(D0-D1) # Vector of critical times
        pmax = P + (S1/(D0-D1))*M*(np.exp(-time*D1) - np.exp(-time*D0))
        pmax[0] = P[0] # Discard stimulus
        # Explicit upper bound for Kon
        K0, K1 = self.param['K0'], self.param['K1']
        sigma = expit(self.basal + pmax @ ((self.inter > 0) * self.inter))
        Kon = (1-sigma)*K0 + sigma*K1 + 1e-10 # Fix precision errors
        Kon[0] = 0 # Ignore stimulus
        return Kon

    def flow(self, time, state):
        """
        Deterministic flow for the bursty model.
        """
        M, P = state['M'], state['P']
        D0, D1, S1 = self.param['D0'], self.param['D1'], self.param['S1']
        # Explicit solution of the ODE generating the flow
        Mnew = M*np.exp(-time*D0)
        Pnew = ((S1/(D0-D1))*M*(np.exp(-time*D1) - np.exp(-time*D0))
                + P*np.exp(-time*D1))
        Mnew[0], Pnew[0] = M[0], P[0] # Discard stimulus
        return Mnew, Pnew

    def step(self):
        """
        Compute the next jump and the next step of the
        thinning method, in the case of the bursty model.
        """
        if self.thin_cst is None:
            # Adaptive thinning parameter
            tau = np.sum(self.kon_bound())
        else: tau = self.thin_cst
        jump = False # Test if the jump is a true or phantom jump
        # 0. Draw the waiting time before the next jump
        U = np.random.exponential(scale=1/tau)
        # 1. Update the continuous states
        M, P = self.flow(U, self.state)
        self.state['M'], self.state['P'] = M, P
        # 2. Compute the next jump
        G = self.basal.size # Genes plus stimulus
        v = self.kon(P)/tau # i = 1, ..., G-1 : burst of mRNA i
        v[0] = 1 - np.sum(v[1:]) # i = 0 : no change (phantom jump)
        i = np.random.choice(G, p=v)
        if i > 0:
            r = self.param['B'][i]
            self.state['M'][i] += np.random.exponential(1/r)
            jump = True
        return U, jump

    def simulation(self, timepoints, verb=False):
        """
        Exact simulation of the network in the bursty PDMP case.
        """
        G = self.basal.size
        types = [('M','float64'), ('P','float64')]
        sim = [] # List of states to be recorded
        c0, c1 = 0, 0 # Jump counts (phantom and true)
        T = 0
        # Core loop for simulation and recording
        Told, state_old = T, self.state.copy()
        for t in timepoints:
            while T < t:
                Told, state_old = T, self.state.copy()
                U, jump = self.step()
                T += U
                if jump: c1 += 1
                else: c0 += 1
            M, P = self.flow(t - Told, state_old)
            sim += [np.array([(M[i],P[i]) for i in range(1,G)], dtype=types)]
        # Update the current state
        self.state['M'], self.state['P'] = M, P
        # Display info about jumps
        if verb:
            ctot = c0 + c1
            if ctot > 0:
                print('Exact simulation used {} jumps '.format(ctot)
                    + 'including {} phantom jumps '.format(c0)
                    + '({:.2f}%)'.format(100*c0/ctot))
            else: print('Exact simulation used no jump')
        return np.array(sim)
