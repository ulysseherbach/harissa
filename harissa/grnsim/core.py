"""Core functions for the simulation in the network model"""

import numpy as np

### Case of the full model (promoters)
def flow_full(time, state, param):
    """Deterministic flow for the full model."""
    E, M, P = state['E'], state['M'], state['P']
    S0, D0, S1, D1 = param['S0'], param['D0'], param['S1'], param['D1']
    ### Scale constants
    A0 = S0/D0 # mRNA scales
    A1 = (S0*S1)/(D0*D1) # Protein scales
    ### Explicit solution of the ODE generating the flow
    Mnew = A0*E + (M - A0*E)*np.exp(-time*D0)
    Pnew = A1*E + (P - A1*E)*np.exp(-time*D1)
    Pnew += (A1*D1/(D0-D1))*(M/A0 - E)*(np.exp(-time*D1) - np.exp(-time*D0))
    return Mnew, Pnew

def step_exact_full(model):
    """Compute the next jump and the next step of the
    thinning method, in the case of the full model."""
    tau = model.thin_cst
    jump = False # Test if the jump is a true or phantom jump
    ### 0. Draw the waiting time before the next jump
    U = np.random.exponential(scale=1/tau)
    ### 1. Update the continuous states
    M, P = flow_full(U, model.state, model.param)
    model.state['M'], model.state['P'] = M, P
    ### 2. Update the promoters
    a, b = model.kon(P)/tau, model.koff(P)/tau
    G, E = model.size, model.state['E']
    v = np.zeros(G+1) # Probabilities for possible transitions
    v[1:] = a*(1-E) + b*E # i = 0, ..., n-1 : switch promoter i
    v[0] = 1 - np.sum(v[1:]) # i = -1 : no change (phantom jump)
    i = np.random.choice(G+1, p=v) - 1
    if (i > -1):
        model.state['E'][i] = 1 - E[i]
        jump = True
    return U, jump

def sim_exact_full(model, timepoints, info=False):
    """Exact simulation of the network in the two-state model case."""
    init_state = model.state.copy() # Save the current state
    G = model.size
    sim = []
    types = [('E','uint8'), ('M','float64'), ('P','float64')]
    c0, c1 = 0, 0 # Jump counts (phantom and true)
    T = 0
    ### The core loop for simulation and recording
    Told, state_old = T, model.state.copy()
    for t in timepoints:
        while (t >= T):
            Told, state_old = T, model.state.copy()
            U, jump = step_exact_full(model)
            T += U
            if jump: c1 += 1
            else: c0 += 1
        E = state_old['E']
        M, P = flow_full(t - Told, state_old, model.param)
        sim += [np.array([(E[i],M[i],P[i]) for i in range(G)], dtype=types)]
    ### Restore the initial state of the input
    model.state = init_state
    ### Display info about jumps
    if info:
        print('Exact simulation used {} jumps, '.format(c0+c1)
            + 'including {} phantom jumps '.format(c0)
            + '({:.2f}%).'.format(100*c0/(c1+c0)))
    return np.array(sim)

### Case of the bursty limit model (no promoters)
def flow_bursty(time, state, param):
    """Deterministic flow for the bursty limit model."""
    M, P = state['M'], state['P']
    D0, S1, D1 = param['D0'], param['S1'], param['D1']
    ### Explicit solution of the ODE generating the flow
    Mnew = M*np.exp(-time*D0)
    Pnew = P*np.exp(-time*D1)
    Pnew += (S1/(D0-D1))*M*(np.exp(-time*D1) - np.exp(-time*D0))
    return Mnew, Pnew

def step_exact_bursty(model):
    """Compute the next jump and the next step of the
    thinning method, in the case of the bursty model."""
    tau = model.thin_cst
    jump = False # Test if the jump is a true or phantom jump
    ### 0. Draw the waiting time before the next jump
    U = np.random.exponential(scale=1/tau)
    ### 1. Update the continuous states
    M, P = flow_bursty(U, model.state, model.param)
    model.state['M'], model.state['P'] = M, P
    ### 2. Update the promoters
    G = model.size
    v = np.zeros(G+1) # Probabilities for possible transitions
    v[1:] = model.kon(P)/tau # i = 0, ..., n-1 : burst of mRNA i
    v[0] = 1 - np.sum(v[1:]) # i = -1 : no change (phantom jump)
    i = np.random.choice(G+1, p=v) - 1
    if (i > -1):
        B = model.koff(P)
        S0 = model.param['S0']
        model.state['M'][i] += np.random.exponential(S0[i]/B[i])
        jump = True
    return U, jump

def sim_exact_bursty(model, timepoints, info=False):
    """Exact simulation of the network in the bursty model case."""
    init_state = model.state.copy() # Save the current state
    G = model.size
    sim = []
    types = [('M','float64'), ('P','float64')]
    c0, c1 = 0, 0 # Jump counts (phantom and true)
    T = 0
    ### The core loop for simulation and recording
    Told, state_old = T, model.state.copy()
    for t in timepoints:
        while (t >= T):
            Told, state_old = T, model.state.copy()
            U, jump = step_exact_bursty(model)
            T += U
            if jump: c1 += 1
            else: c0 += 1
        M, P = flow_bursty(t - Told, state_old, model.param)
        sim += [np.array([(M[i],P[i]) for i in range(G)], dtype=types)]
    ### Restore the initial state of the input
    model.state = init_state
    ### Display info about jumps
    if info:
        print('Exact simulation used {} jumps, '.format(c0+c1)
            + 'including {} phantom jumps '.format(c0)
            + '({:.2f}%).'.format(100*c0/(c1+c0)))
    return np.array(sim)

### Case of the deterministic limit model (promoters fully averaged)
def step_ode(model, dt):
    """Euler step for the deterministic limit model."""
    param = model.param
    S0, D0, S1, D1 = param['S0'], param['D0'], param['S1'], param['D1']
    M, P = model.state['M'], model.state['P']
    a, b = model.kon(P), model.koff(P)
    Mnew = (1 - dt*D0)*M + dt*S0*a/(a+b)
    Pnew = (1 - dt*D1)*P + dt*S1*M
    model.state['M'], model.state['P'] = Mnew, Pnew

def sim_ode(model, timepoints):
    """Simulation of the deterministic limit model, which is relevant when
    promoters/RNA are much faster than proteins.
    1. ODE system involving proteins only
    2. Mean level of mRNA given protein levels"""
    init_state = model.state.copy() # Save the current state
    G = model.size
    dt = model.euler_step
    T = 0
    sim = []
    types = [('M','float64'), ('P','float64')]
    ### The core loop for simulation and recording
    for t in timepoints:
        while (t >= T):
            step_ode(model, dt)
            T += dt
        M, P = model.state['M'], model.state['P']
        sim += [np.array([(M[i],P[i]) for i in range(G)], dtype=types)]
    ### Restore the initial state of the input
    model.state = init_state
    return np.array(sim)