### Harissa demo 1 - Simulating networks ###

### Uncomment for testing before "proper" install
# import sys
# sys.path.append("../")

### Import the relevant packages
import numpy as np
import harissa.grnsim as ns

### Load a 3-gene network example
from harissa.networks import net0
network = ns.load(net0)

### Set the interactions
network.theta[0,0] = 5
network.theta[1,0] = 5
network.theta[2,1] = 5

### Set the initial values
network.state['M'] = 1e-2
network.state['P'] = 5e-2

### Set the steps to record
time = np.linspace(0,100,1000)

### Stochastic PDMP model (general regime)
# This is the fundamental mechanistic model
# It includes promoter states
simu = network.simulate(time, method='exact', info=True)
ns.plotsim(time,simu,'pathPDMP.pdf')
ns.histo(simu['M'], simu['P'], 'histoPDMP.pdf')

### Stochastic PDMP model (bursty regime)
# In this case the thinning constant
# can be drastically decreased
network.regime = 'bursty'
network.thin_cst = np.sum(network.K1)
simu = network.simulate(time, method='exact', info=True)
ns.plotsim(time,simu,'pathPDMPbursty.pdf')
ns.histo(simu['M'], simu['P'], 'histoPDMPbursty.pdf')

## Deterministic counterpart model
# This model is provided for comparison
# but we claim it is not realistic in general
simu = network.simulate(time, method='ode')
ns.plotsim(time,simu,'pathODE.pdf')
ns.histo(simu['M'], simu['P'], 'histoODE.pdf')


