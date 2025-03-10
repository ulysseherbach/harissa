# BurstyBase model (reduced/hybrid/bursty - single gene - no feedback)
import numpy as np
from harissa.models import BurstyBase

# Path of result files
result_path = "../examples/results/bursty_base_"

model = BurstyBase()
model.burst_size = 0.5
model.burst_frequency = 2
model.degradation_rate = 1

# Trajectories
time = np.linspace(0, 10, 1000)
sim = model.simulate(time, init_state=1)

# Distribution
x = np.linspace(0, 3, 1000)
init_state = 1
time = np.linspace(0, 5, 6)
p = model.distribution(x, time, init_state)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,3))
plt.plot(sim.t, sim.x)
plt.xlim(sim.t[0], sim.t[-1])
plt.ylim(0)
fig.savefig(result_path + "traj.pdf", bbox_inches='tight')

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,3))
for k, t in enumerate(time):
    plt.plot(x, p[k], label=f"t = {t}")
    print(f"t = {t} -> non-zeros: {np.count_nonzero(p[k])}")
plt.xlim(x[0], x[-1])
plt.ylim(0, 1)
plt.legend(loc='upper right')
fig.savefig(result_path + "dist.pdf", bbox_inches='tight')
