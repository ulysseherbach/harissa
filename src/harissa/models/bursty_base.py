"""
Reduced/Hybrid/Bursty model for a single gene with no feedback
"""
import numpy as np
from harissa.models._checks import check_time_points, check_init_state

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
        time = check_time_points(time)
        check_init_state(init_state)
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
        return Simulation(time, traj)



    def conditional_distribution(self, x, time, init_state, smooth=0.0):
        """
        """
        pass

    def propagator(self, x, time, init_state, smooth=0.0):
        """
        Alias for `conditional_distribution`.
        """
        return self.conditional_distribution(x, time, init_state, smooth)






# Tests
if __name__ == '__main__':
    model = BurstyBase()
    time = np.linspace(0, 5, 6)
    sim = model.simulate(time)
    print(sim.t)
    print(sim.x)
