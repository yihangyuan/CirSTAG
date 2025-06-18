# CirSTAG

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

To utilize CirSTAG, follow the steps outlined below:

1. copy the `my_utils` folder to the root of the target repo folder

2. In your Python code, import the CirSTAG

`from my_utils.utils import spectral_embedding,SAGMAN`


3. Use the CirSTAG
