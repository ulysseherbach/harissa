# Harissa 🌶

[![PyPI - Version](https://img.shields.io/pypi/v/harissa)](https://pypi.org/project/harissa/)
[![Conda - Version](https://img.shields.io/conda/vn/conda-forge/harissa)](https://anaconda.org/conda-forge/harissa)

This is a Python package for both simulation and inference of gene regulatory networks from single-cell data. Its name comes from ‘HARtree approximation for Inference along with a Stochastic Simulation Algorithm.’ It is implemented in the context of a [mechanistic approach](https://doi.org/10.1186/s12918-017-0487-0) to gene regulatory network inference from single-cell data, based upon an underlying stochastic dynamical model driven by the [transcriptional bursting](https://en.wikipedia.org/wiki/Transcriptional_bursting) phenomenon.

*Main functionalities:*

1. Network inference interpreted as calibration of a dynamical model;
2. Data simulation (typically scRNA-seq) from the same dynamical model.

*Other available tools:*

* Basic GRN visualization (directed graphs with positive or negative edge weights);
* Binarization of scRNA-seq data (using gene-specific thresholds derived from the calibrated dynamical model).

The current version of Harissa has benefited from improvements introduced within [Cardamom](https://github.com/eliasventre/cardamom), which can be seen as an alternative method for the inference part. The two inference methods remain complementary at this stage and may be merged into the same package in the future. They were both evaluated in a [recent benchmark](https://doi.org/10.1371/journal.pcbi.1010962).

## Installation

Harissa can be installed using [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/):

```bash
pip install harissa
```

This command will also check for all required dependencies (see below) and install them if necessary. If the installation is successful, all scripts in the [tests](https://github.com/ulysseherbach/harissa/tree/main/tests) folder should run smoothly (note that `network4.py` must be run before `test_binarize.py`).

## Basic usage

```python
from harissa import NetworkModel
model = NetworkModel()

# Inference
model.fit(data)

# Simulation
sim = model.simulate(time)
```

Here `data` should be a two-dimensional array of single-cell gene expression counts, where each row represents a cell and each column represents a gene, except for the first column, which contains experimental time points. A toy example is:

```python
import numpy as np

data = np.array([
    #t g1 g2 g3
    [0, 4, 1, 0], # Cell 1
    [0, 5, 0, 1], # Cell 2
    [1, 1, 2, 4], # Cell 3
    [1, 2, 0, 8], # Cell 4
    [1, 0, 0, 3], # Cell 5
])
```

The `time` argument for simulations is either a single time or a list of time points. For example, a single-cell trajectory (not available from scRNA-seq) from *t* = 0h to *t* = 10h can be simulated using:

```python
time = np.linspace(0, 10, 1000)
```

The `sim` output stores mRNA and protein levels as attributes `sim.m` and `sim.p`, respectively (each row is a time point and each column is a gene).

## About the data

The inference algorithm specifically exploits time-course data, where single-cell profiling is performed at a number of time points after a stimulus (see [this paper](https://doi.org/10.1371/journal.pcbi.1010962) for an example with real data). Each group of cells collected at the same experimental time *t*<sub>*k*</sub> forms a *snapshot* of the biological heterogeneity at time *t*<sub>*k*</sub>. Due to the destructive nature of the measurement process, successive snapshots are made of different cells. Such data is therefore different from so-called ‘pseudotime’ trajectories, which attempt to reorder cells according to some smoothness hypotheses.

## Tutorial

Please see the [notebooks](https://github.com/ulysseherbach/harissa/tree/main/examples) for introductory examples, or the [tests](https://github.com/ulysseherbach/harissa/tree/main/tests) folder for basic usage scripts. To get an idea of the main features, you can start by running the notebooks in order:

- [Notebook 1](https://github.com/ulysseherbach/harissa/blob/main/examples/notebook1.ipynb): simulate a basic repressilator network with 3 genes;
- [Notebook 2](https://github.com/ulysseherbach/harissa/blob/main/examples/notebook2.ipynb): perform network inference from a small dataset with 4 genes;
- [Notebook 3](https://github.com/ulysseherbach/harissa/blob/main/examples/notebook3.ipynb): compare two branching pathways with 4 genes from both ‘single-cell’ and ‘bulk’ viewpoints.

## Numerical acceleration

```python
# Inference
model.fit(data, use_numba=True)

# Simulation
sim = model.simulate(time, use_numba=True)
```

The `use_numba` option is not activated by default for simulations since it takes time to compile (~8s) but it is then much more efficient (~20 times faster) which is typically suited for large numbers of genes and/or cells.

## Dependencies

The package depends on standard scientific libraries `numpy` and `scipy`. Optionally, it can load `numba` for accelerating the inference procedure (used by default) and the simulation procedure (not used by default). It also depends optionally on `matplotlib` and `networkx` for network visualization.

## Citation

If you use Harissa in your work, please cite [this paper](https://doi.org/10.1007/978-3-031-42697-1_7) (also available on [arXiv](https://doi.org/10.48550/arXiv.2309.05112)).
