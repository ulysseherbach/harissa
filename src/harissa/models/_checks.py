"""
Various checks for function parameters within the API.
"""
import numpy as np

def _check_time_points(time):
    """
    Check time points for trajectory simulations.
    """
    if np.ndim(time) == 0:
        time = np.array([time], dtype=float)
    elif np.ndim(time) == 1:
        time = np.array(time, dtype=float)
    else:
        raise ValueError("Time points should either be scalar or 1D array.")
    if np.any(time != np.sort(time)):
        raise ValueError("Time points must be given in increasing order.")
    if time[0] < 0:
        raise ValueError("Time points must be nonnegative.")
    return time

def _check_init_state(init_state, shape=None):
    """
    Check initial state for trajectory simulations.
    """
    if isinstance(shape, tuple) and (np.shape(init_state) != shape):
        if shape == ():
            raise ValueError("Initial state must be a scalar value.")
        else:
            raise ValueError(f"Initial state must have shape {shape}.")
    if np.any(np.array(init_state) < 0):
        raise ValueError("Initial state must be nonnegative.")
    return init_state


# Tests
if __name__ == '__main__':
    time = np.linspace(0, 10, 100)
    _check_time_points(time)
    state = 0, 1
    _check_init_state(state)
