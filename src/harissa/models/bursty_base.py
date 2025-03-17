"""
Reduced/Hybrid/Bursty model for a single gene with no feedback.
"""
import numpy as np
from scipy.special import hyp1f1
from harissa.models._checks import (_check_time_points, _check_init_state,
    _check_state_array)

class Simulation:
    """
    Basic object to store simulations.
    """
    def __init__(self, t, x):
        self.t = t # Time points
        self.x = x # Proteins or mRNA

class BurstyBase:
    """
    Reduced-Hybrid-Bursty model for a single gene with no feedback.
    NB: For this model the propagator is available in analytical form.
    """
    def __init__(self,
        burst_size=1.0,
        burst_frequency=1.0,
        degradation_rate=1.0):
        # Set model parameters
        self.burst_size = burst_size
        self.burst_frequency = burst_frequency
        self.degradation_rate = degradation_rate

    def simulate(self, time, init_state=0.0, verb=False):
        """
        Exact simulation of the model (extracted at given time points).
        """
        burst_size = self.burst_size
        burst_frequency = self.burst_frequency
        degradation_rate = self.degradation_rate
        # Check parameters
        time = _check_time_points(time).reshape((-1,))
        init_state = _check_init_state(init_state, shape=())
        # Number of bursts
        n = np.random.poisson(burst_frequency * time[-1])
        # Burst times and heights
        t = np.random.uniform(low=0, high=time[-1], size=n)
        h = np.random.exponential(scale=burst_size, size=n)
        # Build the post-jump embedded Markov chain
        t = np.append(0, np.sort(t))
        x = np.zeros(n + 1)
        x[0] = init_state
        for k in range(n):
            dt = t[k+1] - t[k]
            x[k+1] = x[k] * np.exp(- degradation_rate * dt) + h[k]
        # Extract states at user time points
        traj = np.zeros(time.size)
        for i, u in enumerate(time):
            k = np.argwhere(t <= u)[-1][0]
            dt = u - t[k]
            traj[i] = x[k] * np.exp(- degradation_rate * dt)
        if verb:
            print(f'Simulation generated {n} jumps.')
        # Simplify scalar case
        if time.size == 1:
            time = time.reshape(())
            traj = traj.reshape(())
        return Simulation(time, traj)

    def distribution(self, x, x0, time, smooth=1e-5, discrete=False):
        """
        Time-dependent conditional distribution with respect to initial
        state `x0` at time 0, interpreted as a probability kernel p(x|x0)
        for each given time and initial state.
        When `discrete` is set to False (default), returns an array of
        probability density functions evaluated at `x` points.
        When `discrete` is set to True, returns an array of discrete
        probability measures corresponding to `x` points.
        """
        b = 1 / self.burst_size
        k = self.burst_frequency
        d = self.degradation_rate
        # Check and reshape parameters
        t = _check_time_points(time).reshape((-1, 1, 1))
        x0 = _check_state_array(x0).reshape((1, -1, 1))
        x = _check_state_array(x).reshape((1, 1, -1))
        # Smoothing standard deviation
        s = smooth
        # State when no burst occurred
        xt = np.exp(- d * t) * x0
        # Array of translated states
        x = x - xt
        # Scaling factors
        e0 = np.exp(d * t)
        e1 = np.exp(k * t)
        e1[t == 0] = 0 # Discarded
        r1 = b * (e0 - 1)
        r2 = (k/d) * (x >= 0) / (e1 - 1)
        # Probability that no burst occurred
        w = np.exp(- k * t)
        # Pre-burst part of the distribution (regularized)
        h = np.exp(- x**2 / (2 * s**2)) / (np.sqrt(2 * np.pi) * s)
        # Post-burst part of the distribution
        # # Option 1: gamma-mixture formula
        # g = r1 * r2 * np.exp(- b * e0 * x) * hyp1f1(k/d + 1, 2, r1 * x)
        # Option 2: using Kummer's transformation
        g = r1 * r2 * np.exp(- b * x) * hyp1f1(1 - k/d, 2, -r1 * x)
        # Overall distribution
        p = w * h + (1 - w) * g
        # Simplify particular cases
        if (t.size == 1) and (x0.size > 1):
            p = p[0]
        elif (t.size == 1) and (x0.size == 1):
            p = p[0, 0]
        elif (t.size > 1) and (x0.size == 1):
            p = p[:, 0]
        # Normalize in the discrete case
        if discrete:
            p /= np.sum(p, axis=-1, keepdims=True)
        return p


# Tests
if __name__ == '__main__':
    model = BurstyBase()
    # Simulation
    time = np.linspace(0, 50, 6)
    sim = model.simulate(time, verb=True)
    print(sim.t)
    print(sim.x)
    # Distribution
    x = np.linspace(0, 2, 4)
    time = 0, 1, 2
    x0 = 0, 0.1
    p = model.distribution(x, x0, time, smooth=1e-2, discrete=True)
    print(p)
