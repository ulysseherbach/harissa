# BurstyBase model (reduced/hybrid/bursty - single gene - no feedback)
import numpy as np
import matplotlib.pyplot as plt
from harissa.models import BurstyBase

# Path of result files
result_path = "../examples/results/bursty_base_"

model = BurstyBase()
model.burst_size = 0.5
model.burst_frequency = 2.5
model.degradation_rate = 0.8

# Compute a single trajectory
time = np.linspace(0, 10, 1000)
sim = model.simulate(time, init_state=1)

# Show the trajectory
fig = plt.figure(figsize=(12,3))
plt.plot(sim.t, sim.x)
plt.xlim(sim.t[0], sim.t[-1])
plt.ylim(0)
fig.savefig(result_path + "traj.pdf", bbox_inches='tight')

# Time-dependent distribution
x = np.linspace(0, 3, 1000)
init_state = 1
time = np.linspace(0, 5, 6)
p = model.distribution(x, time, init_state)

# Confirm analytical formula
t = time[1]
n_cells = 10000
cell_pop = np.zeros(n_cells)
for k in range(n_cells):
    # print(f"Cell {k+1}...")
    sim = model.simulate(t, init_state)
    cell_pop[k] = sim.x[0]

# Show time-dependent distribution
fig = plt.figure(figsize=(8,3))
for k, t in enumerate(time):
    plt.plot(x, p[k], label=f"t = {t}")
    print(f"t = {t} -> non-zeros: {np.count_nonzero(p[k])}")
plt.hist(cell_pop, density=True, bins=100, color='lightgray', zorder=0)
plt.xlim(x[0], x[-1])
plt.ylim(0, 1)
plt.legend(loc='upper right')
fig.savefig(result_path + "dist.pdf", bbox_inches='tight')
