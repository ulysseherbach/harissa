"""Settings for the inference procedure"""

### Gibbs sampling parameters
sample_size = 1000
iter_gibbs_init = 10
iter_gibbs = 1

### Variational method
var_tol = 1e-5
var_iter_max = 1000

### Penaliztion
penalization = 2*1.5e-2
lasso_mix = 0.9

### Maximization step
m_tol = 1e-5
learning_rate = 1e0
iter_grad = 1000

### Sum-product algorithm
sp_threshold = 0
sp_tol = 1e-5
sp_iter_max = 20
sp_cmax = 5 # Maximum clique size