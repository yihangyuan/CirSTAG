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

```python
    from my_utils.utils import CIRSTAG
```


3. Use the CirSTAG function with your GNN data inputs and outputs:

```python
TopEig, TopEdgeList, TopNodeList, node_score = CIRSTAG(data_input, data_output)
```
`TopEig` This represents the model score, indicating the robustness of the model.
`TopEdgeList` A list of edges ranked from the least robust to the most robust edge.
`TopNodeList` A list of nodes ranked from the least robust to the most robust node.
`node_score` The CirSTAG score for each node, starting from node number 1 up to node number N, where N is the total number of nodes in `data_input`.

4. The `data_input` is n by n csr format adjacency matrix, and `data_output` is n by m output embeddings. If your data is a multidimensional array, make sure to flatten it before use.

5. The `CirSTAG` function has the following default options:
   - `k=10`: Specifies the kNN graph.
   - `num_eigs=2`: Determines the number of general eigenpairs.
   - `sparse=True`: Indicates whether to construct a sparse kNN graph.
   - `weighted=True`: Determines whether to construct a weighted graph.
   - `M=48`: Determines the maximum number of connections each node in the graph can have in the base layer of the HNSW graph.
   - `use_eig=False` if False use SVD else use eigen decomposition
