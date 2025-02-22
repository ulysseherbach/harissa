"""
Perform simulations using the PDMP model - Fast version using Numba
NB: This module takes time to compile (~8s) but is much more efficient,
which is typically suited for large numbers of genes and/or cells
"""
import numpy as np
from numba import njit

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

    def simulation(self, timepoints, verb=False):
        """
        Exact simulation of the network in the bursty PDMP case.
        """
        basal, inter = self.basal, self.inter
        K0, K1, B = self.param['K0'], self.param['K1'], self.param['B']
        D0, D1, S1 = self.param['D0'], self.param['D1'], self.param['S1']
        M, P, thin = self.state['M'], self.state['P'].copy(), self.thin_cst
        if np.size(timepoints) == 1: timepoints = np.array([timepoints])
        timepoints = timepoints.astype(float)
        # Compute the simulation
        states = simulate(timepoints, basal, inter,
            K0, K1, B, D0, D1, S1, M, P, thin, verb)
        # Update the current state
        self.state['M'], self.state['P'] = states[0,-1], states[1,-1]
        # Store the results
        types = [('M','float64'), ('P','float64')]
        G, sim = basal.size, []
        for k, t in enumerate(timepoints):
            M, P = states[0,k], states[1,k]
            sim += [np.array([(M[i],P[i]) for i in range(1,G)], dtype=types)]
        return np.array(sim)

# Core functions for Numba acceleration

@njit
def kon(basal, inter, K0, K1, P):
    """
    Interaction function kon (off->on rate), given protein levels p.
    """
    Phi = np.exp(basal + P @ inter)
    Kon = (K0 + K1*Phi)/(1 + Phi)
    Kon[0] = 0 # Ignore stimulus
    return Kon

@njit
def kon_bound(basal, inter, K0, K1, D0, D1, S1, M, P):
    """
    Compute the current kon upper bound.
    """
    # Explicit upper bound for P
    time = np.log(D0/D1)/(D0-D1) # Vector of critical times
    pmax = P + (S1/(D0-D1))*M*(np.exp(-time*D1) - np.exp(-time*D0))
    pmax[0] = P[0] # Discard stimulus
    # Explicit upper bound for Kon
    Phi = np.exp(basal + pmax @ ((inter > 0) * inter))
    Kon = (K0 + K1*Phi)/(1 + Phi) + 1e-10 # Fix precision errors
    Kon[0] = 0 # Ignore stimulus
    return Kon

@njit
def flow(D0, D1, S1, M, P, time):
    """
    Deterministic flow for the bursty model.
    """
    Mnew = M*np.exp(-time*D0)
    Pnew = ((S1/(D0-D1))*M*(np.exp(-time*D1) - np.exp(-time*D0))
            + P*np.exp(-time*D1))
    Mnew[0], Pnew[0] = M[0], P[0] # Discard stimulus
    state = np.zeros((2,M.size))
    state[0], state[1] = Mnew, Pnew
    return state

@njit
def step(basal, inter, K0, K1, B, D0, D1, S1, M, P, thin):
    """
    Compute the next jump and the next step of the
    thinning method, in the case of the bursty model.
    """
    if thin is None:
        # Adaptive thinning parameter
        tau = np.sum(kon_bound(basal, inter, K0, K1, D0, D1, S1, M, P))
    else: tau = thin
    jump = False # Test if the jump is a true or phantom jump
    # 0. Draw the waiting time before the next jump
    U = np.random.exponential(scale=1/tau)
    # 1. Update the continuous states
    state = flow(D0, D1, S1, M, P, U)
    M[:], P[:] = state[0], state[1]
    # 2. Compute the next jump
    v = kon(basal, inter, K0, K1, P)/tau # i = 1, ..., G-1 : burst of mRNA i
    v[0] = 1 - np.sum(v[1:]) # i = 0 : no change (phantom jump)
    i = np.nonzero(np.random.multinomial(1, v))[0][0]
    if i > 0:
        r = B[i]
        M[i] += np.random.exponential(1/r)
        jump = True
    return U, jump

@njit
def simulate(timepoints, basal, inter, K0, K1, B, D0, D1, S1, M, P,
    thin, verb):
    """
    Exact simulation of the network in the bursty PDMP case.
    """
    states = np.zeros((2,timepoints.size,basal.size))
    # sim = np.zeros(len(timepoints), dtype=types)
    c0, c1 = 0, 0 # Jump counts (phantom and true)
    T = 0
    # Core loop for simulation and recording
    Told, Mold, Pold = T, M.copy(), P.copy()
    for k, t in enumerate(timepoints):
        while T < t:
            Told, Mold, Pold = T, M.copy(), P.copy()
            U, jump = step(basal, inter, K0, K1, B, D0, D1, S1, M, P, thin)
            T += U
            if jump: c1 += 1
            else: c0 += 1
        state = flow(D0, D1, S1, Mold, Pold, t - Told)
        states[0,k], states[1,k] = state[0], state[1]
    # Display info about jumps
    if verb:
        ctot = c0 + c1
        if ctot > 0:
            print(f'Exact simulation used {ctot} jumps '
                + f'including {c0} phantom jumps '
                + f'({int(100*c0/ctot)}%)')
        else: print('Exact simulation used no jump')
    return states
