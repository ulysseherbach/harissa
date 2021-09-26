"""
Perform simulations using the ODE model
"""
import numpy as np

class ApproxODE:
    """
    ODE version of the network model (very rough approximation of the PDMP)
    """
    def __init__(self, a, d, basal, inter):
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
        # Simulation parameter
        d0, d1, s1 = np.max(D0), np.max(D1), np.max(S1)
        k0, k1, b = np.max(K0), np.max(K1), np.max(B)
        self.euler_step = 0.01/np.max([d0, d1, s1, k0, k1, b])

    def kon(self, p):
        """
        Interaction function kon (off->on rate), given protein levels p.
        """
        Phi = np.exp(self.basal + p @ self.inter)
        K0, K1 = self.param['K0'], self.param['K1']
        Kon = (K0 + K1*Phi)/(1 + Phi)
        Kon[0] = 0 # Ignore stimulus
        return Kon

    def step_ode(self, dt):
        """
        Euler step for the deterministic limit model.
        """
        M, P = self.state['M'], self.state['P']
        D0, D1, S1 = self.param['D0'], self.param['D1'], self.param['S1']
        a, b = self.kon(P), self.param['B']
        Mnew = (1 - dt*D0)*M + dt*a/b
        Pnew = (1 - dt*D1)*P + dt*S1*M
        Mnew[0], Pnew[0] = M[0], P[0] # Discard stimulus
        self.state['M'], self.state['P'] = Mnew, Pnew

    def simulation(self, timepoints, verb=False):
        """
        Simulation of the deterministic limit model, which is relevant when
        promoters/RNA are much faster than proteins.
        1. ODE system involving proteins only
        2. Mean level of mRNA given protein levels
        """
        G = self.basal.size
        dt = self.euler_step
        if np.size(timepoints) > 1:
            dt = np.min([dt, np.min(timepoints[1:] - timepoints[:-1])])
        types = [('M','float64'), ('P','float64')]
        sim = []
        T, c = 0, 0
        ### The core loop for simulation and recording
        for t in timepoints:
            while T < t:
                self.step_ode(dt)
                T += dt
                c += 1
            M, P = self.state['M'], self.state['P']
            sim += [np.array([(M[i],P[i]) for i in range(1,G)], dtype=types)]
        # Display info about steps
        if verb:
            if c > 0:
                print('ODE simulation used {} steps '.format(c)
                    + '(step size = {:.5f})'.format(dt))
            else: print('ODE simulation used no step')
        return np.array(sim)
