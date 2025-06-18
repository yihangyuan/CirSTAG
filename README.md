# CirSTAG
Code for CirSTAG — Circuit Stability Analysis on Graph-based Manifolds — is an open-source implementation of the framework we presented at DAC 2025. CirSTAG is a spectral method for circuit-level robustness and stability analysis of graph neural networks. We introduce the CirSTAG score, a distance-mapping-distortion metric that upper-bounds the local Lipschitz constant on a circuit manifold and quantifies a model’s sensitivity to feature or topological perturbations.
![Overview of the CirSTAG](/CirSTAG.png)

## Installation
requirement for both Python and Julia packages

### Python packages

```bash
pip install \
    hnswlib \
    numpy scipy scikit-learn \
    networkx \
    torch\
    torch-sparse \
    julia
```


### Julia packages

Open a Julia REPL and run:

```julia
using Pkg
Pkg.add([
    "SparseArrays", "LinearAlgebra", "Clustering", "NearestNeighbors",
    "Distances", "Laplacians", "Arpack", "Statistics", "DelimitedFiles",
    "StatsBase", "Random", "PyCall", "LightGraphs", "TickTock",
    "MatrixMarket", "Metis", "LinearMaps", "MAT"
])
```

## Usage
**CirSTAG code**

`cd CirSTAG/`

**Intergration Example**

`cd intergration/`

## Citing


```bibtex
@inproceedings{yuan2025cirstag,
  author    = {Wuxinlin Cheng, Yihang Yuan, Chenhui Deng, Ali Aghdaei, Zhiru Zhang and Zhuo Feng},
  title     = {{CirSTAG}: Circuit Stability Analysis on Graph-based Manifolds},
  booktitle = {Proceedings of the 62nd Design Automation Conference (DAC)},
  year      = {2025}
}
```

## License

CirSTAG is released under the **MIT License**.  
See \`LICENSE\` for the full text.