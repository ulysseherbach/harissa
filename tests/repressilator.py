# Basic repressilator network (3 genes)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from harissa import NetworkModel

# Path of result files
result_path = "../examples/results/"

# Model
model = NetworkModel(3)
model.d[0] = 1
model.d[1] = 0.2
model.basal[1] = 5
model.basal[2] = 5
model.basal[3] = 5
model.inter[1,2] = -10
model.inter[2,3] = -10
model.inter[3,1] = -10

# Time points
time = np.linspace(0,100,1000)

# Simulation of the PDMP model
sim = model.simulate(time)

# Simulation of the ODE model (slow-fast limit)
sim_ode = model.simulate_ode(time, P0=[0,0,0.1,0.2])

# Figure
fig = plt.figure(figsize=(10,6))
gs = gridspec.GridSpec(3, 1, hspace=0.6)

# Plot mRNA levels
ax = plt.subplot(gs[0,0])
for i in range(3):
    ax.set_title(f"mRNA levels ($d_0 = {model.d[0].mean()}$)")
    ax.plot(sim.t, sim.m[:,i], label=f"$M_{{{i+1}}}$")
    ax.set_xlim(sim.t[0], sim.t[-1])
    ax.set_ylim(0, 1.2*np.max(sim.m))
    ax.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)

# Plot protein levels
ax = plt.subplot(gs[1,0])
for i in range(3):
    ax.set_title(f"Protein levels ($d_1 = {model.d[1].mean()}$)")
    ax.plot(sim.t, sim.p[:,i], label=f"$P_{{{i+1}}}$")
    ax.set_xlim(sim.t[0], sim.t[-1])
    ax.set_ylim(0, np.max([1.2*np.max(sim.p), 1]))
    ax.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)

# Plot protein levels (ODE model)
ax = plt.subplot(gs[2,0])
for i in range(3):
    ax.set_title(r"Protein levels - ODE model ($d_0/d_1\to\infty$)")
    ax.plot(sim_ode.t, sim_ode.p[:,i], label=f"$P_{{{i+1}}}$")
    ax.set_xlim(sim_ode.t[0], sim_ode.t[-1])
    ax.set_ylim(0,1)
    ax.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)

fig.savefig(result_path + "repressilator.pdf", bbox_inches='tight')
