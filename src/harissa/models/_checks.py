"""
Various checks for function parameters within the API.
"""
import numpy as np

def check_time_points(time):
    """
    Check time points for trajectory simulations.
    """
    if np.shape(time) == ():
        time = np.array([time], dtype=float)
    else:
        time = np.array(time, dtype=float)
    if np.any(time != np.sort(time)):
        raise ValueError("Time points must be given in increasing order.")
    if time[0] < 0:
        raise ValueError("Time points must be nonnegative.")
    return time

def check_init_state(state):
    """
    Check initial state for trajectory simulations.
    """
    if np.any(np.array(state) < 0):
        raise ValueError("Initial state must be nonnegative.")
    return state


# Tests
if __name__ == '__main__':
    time = np.linspace(0, 10, 100)
    check_time_points(time)
    state = 0, 1
    check_init_state(state)
