"""
Various checks for function parameters within the API.
"""
import numpy as np

def _check_time_points(time):
    """
    Check time points for trajectory simulations or distributions.
    """
    time = np.array(time, dtype=float, ndmin=1)
    if np.ndim(time) > 1:
        raise ValueError("Time points should either be scalar or 1D array.")
    if np.any(time != np.sort(time)):
        raise ValueError("Time points must be given in increasing order.")
    if np.any(time < 0):
        raise ValueError("Time points must be nonnegative.")
    return time

def _check_init_state(init_state, shape=None):
    """
    Check initial state for trajectory simulations.
    """
    init_state = np.array(init_state, dtype=float)
    if np.any(init_state < 0):
        raise ValueError("Initial state must be nonnegative.")
    # Check for particular shape
    if (shape is not None) and (np.shape(init_state) != shape):
        if shape == ():
            raise ValueError("Initial state must be a scalar value.")
        else:
            raise ValueError(f"Initial state must have shape {shape}.")
    return init_state

def _check_state_array(state):
    """
    Check array of scalar states for distributions.
    """
    state = np.array(state, dtype=float)
    if np.ndim(state) > 1:
        raise ValueError("States should either be scalar or 1D array.")
    if np.any(state < 0):
        raise ValueError("States must be nonnegative.")
    return state

# Tests
if __name__ == '__main__':
    time = np.linspace(0, 10, 100)
    _check_time_points(time)
    init_state = 0, 1
    _check_init_state(init_state)
    state = 0, 1
    _check_state_array(state)
