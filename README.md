# Harissa

This is a Python package for both simulation and reverse engineering of gene regulatory networks from single-cell data. Its name comes from ‘HARtree approximation for Inference along with a Stochastic Simulation Algorithm.’ It was implemented in the context of a [mechanistic approach to gene regulatory network inference from single-cell data](https://bmcsystbiol.biomedcentral.com/articles/10.1186/s12918-017-0487-0) and is based upon an underlying stochastic dynamical model driven by the transcriptional bursting phenomenon.

*Main functionalities:*

1. Network inference interpreted as calibration of a dynamical model;
2. Data simulation (typically scRNA-seq) from the same dynamical model.

*Other available tools:*

* Basic GRN visualization (directed graphs with positive or negative edge weights);
* Binarization of scRNA-seq data (using gene-specific thresholds derived from the data-calibrated dynamical model).

### Tutorial

Please see the [harissa-notebooks](https://github.com/ulysseherbach/harissa-notebooks) for introductory examples, or the `tests` folder for basic usage scripts.

### Dependencies

The package depends on the standard scientific libraries `numpy` and `scipy`. Optionally, it loads `numba` for accelerating the inference procedure (used by default) and the simulation procedure (not used by default). It also depends optionally on `matplotlib` and `networkx` which are required for network visualization.