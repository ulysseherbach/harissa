### Inferring networks using the automodel package ###
import sys
sys.path.append('../')
import numpy as np
import harissa.automodel as am

### Load the data
data = np.loadtxt('data.txt')
### Number of genes
G = np.size(data[0])
### Hyperparameter values
a = np.array([G*[0.2], G*[2], G*[0.5]])
c = np.array(5*[1])

### Initial theta value
theta0 = am.neutral_theta(a, c)

### Penalization strength
# am.config.penalization = 3e-2

### Inference
res = am.infer(data, theta0, a, c, nsteps=10, traj_theta=True)

### Plot the trajectory of theta along the steps
res.traj_theta.plot('test_vtheta.pdf')

### Inferred theta value
print(res.theta)