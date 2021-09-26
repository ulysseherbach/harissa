# Harissa

---

Warning: this repository is still under construction

---

### Tools for mechanistic-based gene network inference
This is a Python package for inferring gene regulatory networks from single-cell data. Its name comes from ‘HARtree approximation for Inference along with a Stochastic Simulation Algorithm.’ It was implemented in the context of a [mechanistic approach to gene regulatory network inference from single-cell data](https://bmcsystbiol.biomedcentral.com/articles/10.1186/s12918-017-0487-0).

### Tutorial and examples
Please see the [examples](https://github.com/ulysseherbach/harissa/tree/master/examples) folder.

### Dependencies
The package depends on the following standard scientific libraries: `numpy`, `scipy`. Besides, it loads `numba` for accelerating the inference procedure (optional, activated by default).
