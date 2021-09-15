# Basic repressilator network (3 genes)
import sys; sys.path += ['../']
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from harissa import NetworkModel

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

time = np.linspace(0,200,1000)
sim = model.simulate(time)

# Figure
fig = plt.figure(figsize=(12,4))
gs = gridspec.GridSpec(2, 1)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])

# Plot proteins
for i in range(3):
    ax1.plot(sim.t, sim.p[:,i], label=r'$P_{}$'.format(i+1))
    ax1.set_xlim(sim.t[0], sim.t[-1])
    ax1.set_ylim(0, np.max([1.2*np.max(sim.p), 1]))
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='y', left=False, labelleft=False)
    ax1.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)

# Plot mRNA
for i in range(3):
    ax2.plot(sim.t, sim.m[:,i], label=r'$M_{}$'.format(i+1))
    ax2.set_xlim(sim.t[0], sim.t[-1])
    ax2.set_ylim(0, 1.2*np.max(sim.m))
    ax2.tick_params(axis='y', left=False, labelleft=False)
    ax2.legend(loc='upper left', ncol=4, borderaxespad=0, frameon=False)

fig.savefig('repressilator.pdf', bbox_inches='tight')
