# Harissa

This is a Python package for both simulation and inference of gene regulatory networks from single-cell data. Its name comes from ‘HARtree approximation for Inference along with a Stochastic Simulation Algorithm.’ It was implemented in the context of a [mechanistic approach](https://doi.org/10.1186/s12918-017-0487-0) to gene regulatory network inference from single-cell data, based upon an underlying stochastic dynamical model driven by the [transcriptional bursting](https://en.wikipedia.org/wiki/Transcriptional_bursting) phenomenon.

*Main functionalities:*

1. Network inference interpreted as calibration of a dynamical model;
2. Data simulation (typically scRNA-seq) from the same dynamical model.

*Other available tools:*

* Basic GRN visualization (directed graphs with positive or negative edge weights);
* Binarization of scRNA-seq data (using gene-specific thresholds derived from the calibrated dynamical model).

The current version of Harissa has benefited from improvements introduced within [Cardamom](https://github.com/eliasventre/cardamom), which can be seen as an alternative method for the inference part. The two inference methods remain complementary at this stage and may be merged into the same package in the future. They were both evaluated in a [recent benchmark](https://doi.org/10.1371/journal.pcbi.1010962).

### Basic usage

```python
from harissa import NetworkModel
model = NetworkModel()

# Inference
model.fit(data)

# Simulation
sim = model.simulate(time)
```

### Tutorial

Please see the [notebooks](https://github.com/ulysseherbach/harissa/tree/main/notebooks) for introductory examples, or the [tests](https://github.com/ulysseherbach/harissa/tree/main/tests) folder for basic usage scripts.

### Dependencies

The package depends on standard scientific libraries `numpy` and `scipy`. Optionally, it can load `numba` for accelerating the inference procedure (used by default) and the simulation procedure (not used by default). It also depends optionally on `matplotlib` and `networkx` for network visualization.