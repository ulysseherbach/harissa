### Inferring networks using the automodel package ###
import sys; sys.path.append('../')
import numpy as np
from harissa import automodel

### Load the data example (5 genes)
data = np.loadtxt('data.txt')

### Hyperparameter values
G = np.size(data[0]) # Number of genes
a = np.array([G*[0.2], G*[2], G*[0.5]])
c = np.array(G*[1])

### Initial theta value for the inference
theta0 = automodel.neutral_theta(a, c)

### Penalization strength
# am.config.penalization = 3e-2

### Inference
res = automodel.infer(data, theta0, a, c, nsteps=10)

### Inferred theta value
print(res.theta)