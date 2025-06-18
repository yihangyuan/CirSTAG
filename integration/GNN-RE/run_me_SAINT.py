import torch
import pandas as pd
# import random
# import time
import os
# import copy
import pickle
import argparse

import torch_geometric.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from my_utils.utils import spectral_embedding,spectral_embedding_eig,CIRSTAG
from scipy.sparse import coo_matrix, csr_matrix
from torch_sparse import SparseTensor

## new imports
import sys
sys.path.append('/home/yihang/git/GNN-RE/GraphSAINT')
from graphsaint.globals import *
from graphsaint.pytorch_version.models_inference import GraphSAINT
from graphsaint.pytorch_version.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from graphsaint.pytorch_version.utils import *

def adj_torchSparse2scipy(sparse_tensor: SparseTensor) -> csr_matrix:

    row, col, value = sparse_tensor.coo()
    
    # Convert to numpy arrays
    row = row.cpu().numpy()
    col = col.cpu().numpy()
    if value is not None:
        value = value.cpu().numpy()
    else:
        value = np.ones(row.shape[0], dtype=np.float32)

    # Determine the shape of the matrix
    size = sparse_tensor.size(0)
    shape = (size, size)

    # Create a SciPy COO-format sparse matrix
    coo_mat = coo_matrix((value, (row, col)), shape=shape)

    # Optionally convert to CSR format
    csr_mat = coo_mat.tocsr()

    return csr_mat

def prepare(train_data,train_params,arch_gcn):
    """
    Prepare some data structure and initialize model / minibatch handler before
    the actual iterative training taking place.
    """
    adj_full, adj_train, adj_test, feat_full, class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_test = adj_test.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    num_classes = class_arr.shape[1]

    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    printf("TOTAL NUM OF PARAMS = {}".format(sum(p.numel() for p in model.parameters())), style="yellow")
    minibatch_eval=Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)
    model_eval=GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    if args_global.gpu >= 0:
        model = model.cuda()
    return model, minibatch, minibatch_eval, model_eval, adj_full, adj_test

def evaluate_full_batch(model, minibatch, mode='val'):
    """
    Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes
        (e.g., those belonging to the val / test sets).
    """
    loss,preds,labels,embedding = model.eval_step(*minibatch.one_batch(mode=mode))
    if mode == 'val':
        node_target = [minibatch.node_val]
    elif mode == 'test':
        node_target = [minibatch.node_test]
    else:
        assert mode == 'valtest'
        node_target = [minibatch.node_val, minibatch.node_test]
        
    f1mic, f1mac = [], []
    for n in node_target:
        f1_scores = calc_f1(to_numpy(labels[n]), to_numpy(preds[n]), model.sigmoid_loss)
        f1mic.append(f1_scores[0])
        f1mac.append(f1_scores[1])
    f1mic = f1mic[0] if len(f1mic)==1 else f1mic
    f1mac = f1mac[0] if len(f1mac)==1 else f1mac
    
    # f1mic, f1mac = [], []
 
    # f1_scores = calc_f1(to_numpy(labels), to_numpy(preds), model.sigmoid_loss)
    # f1mic.append(f1_scores[0])
    # f1mac.append(f1_scores[1])
    # f1mic = f1mic[0] if len(f1mic)==1 else f1mic
    # f1mac = f1mac[0] if len(f1mac)==1 else f1mac
    
    return loss, f1mic, f1mac, embedding # this is all embeddings

def parse_n_prepare(flags):
    with open(flags.train_config) as f_train_config:
        train_config = yaml.safe_load(f_train_config)
    arch_gcn = {
        'dim': -1,
        'aggr': 'concat',
        'loss': 'softmax',
        'arch': '1',
        'act': 'I',
        'bias': 'norm'
    }
    arch_gcn.update(train_config['network'][0])
    train_params = {
        'lr': 0.01,
        'weight_decay': 0.,
        'norm_loss': True,
        'norm_aggr': True,
        'q_threshold': 50,
        'q_offset': 0
    }
    train_params.update(train_config['params'][0])
    train_phases = train_config['phase']
    for ph in train_phases:
        assert 'end' in ph
        assert 'sampler' in ph
    print("Loading training data..")
    temp_data = load_data(flags.data_prefix)
    train_data = process_graph_data(*temp_data)
    print("Done loading training data..")
    return train_params,train_phases,train_data,arch_gcn

def load_data(prefix, normalize=True):
    """
    Load the various data files residing in the `prefix` directory.
    Files to be loaded:
        adj_full.npz        sparse matrix in CSR format, stored as scipy.sparse.csr_matrix
                            The shape is N by N. Non-zeros in the matrix correspond to all
                            the edges in the full graph. It doesn't matter if the two nodes
                            connected by an edge are training, validation or test nodes.
                            For unweighted graph, the non-zeros are all 1.
        adj_train.npz       sparse matrix in CSR format, stored as a scipy.sparse.csr_matrix
                            The shape is also N by N. However, non-zeros in the matrix only
                            correspond to edges connecting two training nodes. The graph
                            sampler only picks nodes/edges from this adj_train, not adj_full.
                            Therefore, neither the attribute information nor the structural
                            information are revealed during training. Also, note that only
                            a x N rows and cols of adj_train contains non-zeros. For
                            unweighted graph, the non-zeros are all 1.
        role.json           a dict of three keys. Key 'tr' corresponds to the list of all
                              'tr':     list of all training node indices
                              'va':     list of all validation node indices
                              'te':     list of all test node indices
                            Note that in the raw data, nodes may have string-type ID. You
                            need to re-assign numerical ID (0 to N-1) to the nodes, so that
                            you can index into the matrices of adj, features and class labels.
        class_map.json      a dict of length N. Each key is a node index, and each value is
                            either a length C binary list (for multi-class classification)
                            or an integer scalar (0 to C-1, for single-class classification).
        feats.npz           a numpy array of shape N by F. Row i corresponds to the attribute
                            vector of node i.

    Inputs:
        prefix              string, directory containing the above graph related files
        normalize           bool, whether or not to normalize the node features

    Outputs:
        adj_full            scipy sparse CSR (shape N x N, |E| non-zeros), the adj matrix of
                            the full graph, with N being total num of train + val + test nodes.
        adj_train           scipy sparse CSR (shape N x N, |E'| non-zeros), the adj matrix of
                            the training graph. While the shape is the same as adj_full, the
                            rows/cols corresponding to val/test nodes in adj_train are all-zero.
        feats               np array (shape N x f), the node feature matrix, with f being the
                            length of each node feature vector.
        class_map           dict, where key is the node ID and value is the classes this node
                            belongs to.
        role                dict, where keys are: 'tr' for train, 'va' for validation and 'te'
                            for test nodes. The value is the list of IDs of nodes belonging to
                            the train/val/test sets.
    """
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix)).astype(bool)
    adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(prefix)).astype(bool)
    adj_test = scipy.sparse.load_npz('./{}/adj_test.npz'.format(prefix)).astype(bool)
    
    role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, adj_test, feats, class_map, role

def process_graph_data(adj_full, adj_train, adj_test, feats, class_map, role):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list): # check if multi-class
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else: # single-class
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, adj_test, feats, class_arr, role

def inference(model, minibatch_eval, model_eval, adj_test):
    # print("dir_log: ", args_global.dir_log)
    path_saver = '{}/pytorch_models/saved_model_2024-04-08 23-39-59.pkl'.format(args_global.dir_log) # interconnected_modulus
    # path_saver = '{}/pytorch_models/saved_model_2024-04-27 17-55-09.pkl'.format(args_global.dir_log) # add_mul_mux
    # path_saver = '{}/pytorch_models/saved_model_{}.pkl'.format(args_global.dir_log, os.path.basename(args_global.data_prefix)) # add_mul_mux
    model.load_state_dict(torch.load(path_saver))
    model_eval= model
    test_nodes = np.array(list(set(adj_test.nonzero()[0])))
    
    
    # embedding = model.inference_embedding(*minibatch.one_batch(mode='valtest'))
    loss, f1mic_both, f1mac_both, embedding = evaluate_full_batch(model_eval, minibatch_eval, mode='test')
    test_embedding = embedding[test_nodes]
    
    printf("Full validation: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}"\
            .format(f1mic_both, f1mac_both), style='red')
    
    return test_embedding


if __name__ == "__main__":
    # log_dir(args_global.train_config, args_global.data_prefix, git_branch, git_rev, timestamp)
    save_path =  os.path.join('stability_ranks/', os.path.basename(args_global.data_prefix))
    train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global)
    
    if 'eval_val_every' not in train_params:
        train_params['eval_val_every'] = EVAL_VAL_EVERY_EP
     
    model, minibatch, minibatch_eval, model_eval, adj_full, adj_test = prepare(train_data, train_params, arch_gcn)
    
    test_nodes = np.array(list(set(adj_test.nonzero()[0])))
    assert test_nodes[0] == 0
    assert test_nodes[-1] == len(test_nodes) - 1
    endnodes = test_nodes[-1] + 1
    adj_test = adj_test[0:endnodes, 0:endnodes]
    embedding = inference(model, minibatch, model_eval, adj_test)
    TopEig, TopEdgeList, TopNodeList, nodeScore = CIRSTAG(adj_test, embedding.cpu().numpy(), k=50, num_eigs=2,weighted=True,sparse=True, use_eig=True, M=48)
  
    

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path,'CIRSTAG_Rank_EdgeList.pkl'), 'wb') as f:
        pickle.dump(TopEdgeList, f)
    with open(os.path.join(save_path,'CIRSTAG_Rank_NodeList.pkl'), 'wb') as f:
        pickle.dump(TopNodeList, f)
    print("CIRSTAG Rank EdgeList and NodeList saved to: ", save_path)














