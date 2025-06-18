import hnswlib
import numpy as np
from scipy.sparse import coo_matrix, diags, identity, csr_matrix, find
from julia.api import Julia
import scipy.sparse.linalg as sla
import networkx as nx
#import grass_mtx as mtx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from torch_sparse import SparseTensor
import torch
import random
import copy
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian
from scipy.sparse import find, triu
import os
import subprocess

def SPF(adj, L, ICr=0.11):
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./my_utils/SPF.jl")
    agj_c = Main.SPF(adj, L, ICr)

    return agj_c

def julia_eigs(l_in, l_out, num_eigs):
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./my_utils/eigen.jl")
    print('Generate eigenpairs')
    eigenvalues, eigenvectors = Main.main(l_in, l_out, num_eigs)

    return eigenvalues.real, eigenvectors.real

def GetRiemannianDist(Lx, Ly, num_eigs):
    # Gy not updated 
    Lx = Lx.asfptype()
    Ly = Ly.asfptype()
    Dxy, Uxy = julia_eigs(Lx, Ly, num_eigs)
    num_node_tot = Uxy.shape[0]
    TopEig=max(Dxy)
    NodeDegree=Lx.diagonal()

    laplacian_upper = triu(Lx, k=1)# k=1 excludes the diagonal
    rows, cols, _ = find(laplacian_upper)# Find the indices of non-zero elements
    num_edge_tot = len(rows)# Number of total edges
    Zpq = np.zeros((num_edge_tot,))# Initialize edge embedding distance array
    p = rows# one end node of each edge
    q = cols# another end node of each edge

    for i in np.arange(0,num_eigs):
        Zpq = Zpq + np.power(Uxy[p,i]-Uxy[q,i], 2)*Dxy[i]
    Zpq = Zpq/max(Zpq)

    node_score=np.zeros((num_node_tot,))        
    for i in np.arange(0,num_edge_tot):
        node_score[p[i]]=node_score[p[i]]+Zpq[i]
        node_score[q[i]]=node_score[q[i]]+Zpq[i]
    node_score=node_score/NodeDegree
    node_score=node_score/np.amax(node_score)

    TopNodeList = np.flip(node_score.argsort(axis=0))
    TopEdgeList=np.column_stack((p,q))[np.flip(Zpq.argsort(axis=0)),:]

    return TopEig, TopEdgeList, TopNodeList, node_score

def construct_adj(neighs, weight):
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1

    idx0 = np.asarray(list(range(dim)))
    idx1 = neighs[:,0]
    mismatch_idx = ~np.isclose(idx0, idx1, rtol=1e-6)
    neighs[mismatch_idx, 1:] = neighs[mismatch_idx, :k]
    row = (np.repeat(idx0.reshape(-1,1), k, axis=1)).reshape(-1,)
    col = neighs[:,1:].reshape(-1,)
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    data = np.ones(all_row.shape[0])
    adj = csr_matrix((data, (all_row, all_col)), shape=(dim, dim))
    adj.data[:] = 1

    return adj


def construct_weighted_adj(neighs, distances):
    dim = neighs.shape[0]
    k = neighs.shape[1] - 1
    weights = np.exp(-distances)

    idx0 = np.asarray(list(range(dim)))
    idx1 = neighs[:,0]
    mismatch_idx = ~np.isclose(idx0, idx1, rtol=1e-6)
    neighs[mismatch_idx, 1:] = neighs[mismatch_idx, :k]
    row = (np.repeat(idx0.reshape(-1,1), k, axis=1)).reshape(-1,)
    col = neighs[:,1:].reshape(-1,)
    # calculate weights for each edge
    edge_weights = weights[:,1:].reshape(-1,)
    all_row = np.concatenate((row, col), axis=0)
    all_col = np.concatenate((col, row), axis=0)
    all_data = np.concatenate((edge_weights, edge_weights), axis=0)  # use weights instead of ones
    adj = csr_matrix((all_data, (all_row, all_col)), shape=(dim, dim))

    return adj

def hnsw(features, k=10, ef=200, M=48):
    num_samples, dim = features.shape

    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=ef, M=M)
    labels_index = np.arange(num_samples)
    p.add_items(features, labels_index)
    p.set_ef(ef)

    neighs, weight = p.knn_query(features, k+1)
  
    return neighs, weight

def to_unweighted_csr(adj_matrix_csr):
    # Get the indices and indptr from the original matrix
    indices = adj_matrix_csr.indices
    indptr = adj_matrix_csr.indptr
    # Create an array of 1s for the values (all edges have weight 1)
    unweighted_data = np.ones(len(indices), dtype=np.float64)
    # Create the unweighted adjacency matrix in CSR format
    unweighted_adj_matrix = csr_matrix((unweighted_data, indices, indptr), shape=adj_matrix_csr.shape)

    return unweighted_adj_matrix

def write_src_dst_to_mtx(filename, src_nodes, dst_nodes):
    with open(filename, "w") as f:

        for src, dst in zip(src_nodes, dst_nodes):
            f.write(f"{src + 1} {dst + 1}\n")

def write_matrix(filename, matrix):
    # Convert to COO format
    matrix = matrix.tocoo()
    matrix = triu(matrix, 1)
    
    with open(filename, 'w') as f:
        # Write each entry in 1-based indexing
        for row, col, data in zip(matrix.row, matrix.col, matrix.data):
            f.write(f"{row } {col} {data}\n") 
            
            
def CIRSTAG(data_input, data_output, k=10, num_eigs=2,weighted=True,sparse=True, M=48, use_eig=False):
    
    if use_eig:
        data_input = spectral_embedding_eig(data_input, None,use_feature=False,embedding_norm=None,adj_norm=True,eig_julia=False)
    else:
        data_input = spectral_embedding(data_input, None,use_feature=False,embedding_norm=None,adj_norm=True)
    
    neighs_in, distance_in = hnsw(data_input, k,M=M)
    neighs_out, distance_out = hnsw(data_output, k,M=M)
    
    if weighted:
        
        adj_in = construct_weighted_adj(neighs_in, distance_in)
        adj_out = construct_weighted_adj(neighs_out, distance_out)
    else:
        
        adj_in, _, _ = construct_adj(neighs_in, distance_in)
        adj_out, _, _ = construct_adj(neighs_out, distance_out)

    print("adj matrix size before SPF: adj_in: ", adj_in.shape, " adj_out: ", adj_out.shape)
    if sparse:
        adj_in = SPF(adj_in, 4)
        adj_out = SPF(adj_out, 4)
    print("adj matrix size after SPF: adj_in: ", adj_in.shape, " adj_out: ", adj_out.shape)
    print("create laplacian")
    L_in = laplacian(adj_in, normed=False)
    L_out = laplacian(adj_out, normed=False)
    #.tocsr()#adj2laplacian(adj_out)

    print("GetRiemannianDist stage")
    TopEig, TopEdgeList, TopNodeList, node_score = GetRiemannianDist(L_in, L_out, num_eigs)# full function
    
    return TopEig, TopEdgeList, TopNodeList, node_score


def normal_adj(adj):
    adj = SparseTensor.from_scipy(adj)
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0
    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)

    return DAD.to_scipy(layout='csr')

def embedding_normalize(embedding, norm):
    if norm == "unit_vector":
        return normalize(embedding, axis=1)
    elif norm == "standardize":
        scaler = StandardScaler()
        return scaler.fit_transform(embedding)
    elif norm == "minmax":
        scaler = MinMaxScaler()
        return scaler.fit_transform(embedding)
    else:
        return embedding


def spectral_embedding(adj_mtx,features,use_feature=True,embedding_norm=None,adj_norm=True):
    adj_mtx = adj_mtx.asfptype()
    num_nodes = adj_mtx.shape[0]
    if adj_norm:
        adj_mtx = normal_adj(adj_mtx)
    U, S, Vt = svds(adj_mtx, 50)

    spec_embed = np.sqrt(S.reshape(1,-1))*U
    spec_embed = embedding_normalize(spec_embed, embedding_norm)
    if use_feature:
        feat_embed = adj_mtx @ (adj_mtx @ features)/2
        feat_embed = embedding_normalize(feat_embed, embedding_norm)
        spec_embed = np.concatenate((spec_embed, feat_embed), axis=1)
    return spec_embed


def spectral_embedding_eig(adj_mtx,features,use_feature=True,embedding_norm=None,adj_norm=True,eig_julia=False):
    adj_mtx = adj_mtx.asfptype()
    num_nodes = adj_mtx.shape[0]
    if adj_norm:
        adj_mtx = normal_adj(adj_mtx)
    #U, S, Vt = svds(adj_mtx, 50)
    L_mtx = adj2laplacian(adj_mtx)
    
    if not eig_julia:
        S, U = eigsh(L_mtx,k=50,which='SM', maxiter=500000)
    else:
        jl = Julia(compiled_modules=False)
        from julia import Main
        Main.include("./my_utils/eigen.jl")
        S, U = Main.not_main(L_mtx.tocoo(), 50)

    spec_embed = U[:, 1:]
    spec_embed = embedding_normalize(spec_embed, embedding_norm)
    if use_feature:
        feat_embed = adj_mtx @ (adj_mtx @ features)/2
        feat_embed = embedding_normalize(feat_embed, embedding_norm)
        spec_embed = np.concatenate((spec_embed, feat_embed), axis=1)
    return spec_embed

def adj2laplacian(A):
    D = diags(np.squeeze(np.asarray(A.sum(axis=1))), 0)
    L = D - A + identity(A.shape[0]).multiply(1e-6)

    return L