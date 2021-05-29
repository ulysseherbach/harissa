### Simulating networks ###
import sys
sys.path.append("../")
import numpy as np
import harissa.grnsim as ns
### Load a 3-gene example parameter set
import net0

### Define a full model including promoter states
network1 = ns.networks.load(net0, mode='full')
### Set the initial values
network1.state['M'] = 1e-2
network1.state['P'] = 5e-2

### Same but with the bursty limit model
network2 = ns.networks.load(net0, mode='bursty')
network2.state['M'] = 1e-2
network2.state['P'] = 5e-2

### Set the steps to record
time = np.linspace(0,100,1000)

### Stochastic PDMP model (general regime)
# This is the fundamental mechanistic model
# It includes promoter states
simu = network1.simulate(time, method='exact', info=True)
ns.plotsim(time,simu,'pathPDMP.pdf')
ns.histo(simu['M'], simu['P'], 'histoPDMP.pdf')

### Stochastic PDMP model (bursty regime)
# In this case the simulation speed
# can be drastically decreased
simu = network2.simulate(time, method='exact', info=True)
ns.plotsim(time,simu,'pathPDMPbursty.pdf')
ns.histo(simu['M'], simu['P'], 'histoPDMPbursty.pdf')

## Deterministic counterpart model
# This model is provided for comparison
# but we claim it is not realistic in general
simu = network1.simulate(time, method='ode')
ns.plotsim(time,simu,'pathODE.pdf')
ns.histo(simu['M'], simu['P'], 'histoODE.pdf')