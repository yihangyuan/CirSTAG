import torch
import pandas as pd
import random
# import time
# import os
import copy
import pickle
import argparse

import torch_geometric.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from my_utils.utils import spectral_embedding,spectral_embedding_eig,SAGMAN
from scipy.sparse import coo_matrix, csr_matrix
from torch_sparse import SparseTensor
from scipy.sparse.csgraph import connected_components
## new imports
import sys
sys.path.append('/home/yihang/git/GNN-RE/GraphSAINT')
from graphsaint.globals import *
from graphsaint.pytorch_version.models_inference import GraphSAINT
from graphsaint.pytorch_version.minibatch import Minibatch
from graphsaint.utils import *
from graphsaint.metric import *
from graphsaint.pytorch_version.utils import *
from torch.nn.functional import cosine_similarity
# prefix = '../Netlist_to_graph/Graphs_datasets/Interconnected-Modules'
embeddings = "test !"
def apply_perturbation(adj_full:csr_matrix, adj_full_directed: csr_matrix, perturb_rate=0.1) -> tuple[csr_matrix, csr_matrix]:
    
    load_path =  os.path.join('stability_ranks/', os.path.basename(args_global.data_prefix), 'SAGMAN_Rank_NodeList.pkl')
    with open(load_path, 'rb') as f:
        NodeList = pickle.load(f)
    
    length = len(NodeList)
    perturb_len = int(length * perturb_rate)
    unstable_nodes_list = NodeList[:perturb_len]
    stable_nodes_list = NodeList[-perturb_len:]
    assert len(unstable_nodes_list) == len(stable_nodes_list)
    
    ############# unstable nodes #############
    input_num_dict = {}   
    for i in unstable_nodes_list:
        rows, _ = adj_full_directed[:, i].nonzero() # the column represent the nodes that points to i
        if i in rows: # skip self-loop/ PO
            continue
        num_of_inputs = len(rows)
        input_num_dict.setdefault(num_of_inputs, []).append(i)
        
    data_unstable = adj_full.toarray()

    for input_num, input_list in input_num_dict.items(): # iterate over the number of inputs
        num_of_pairs = int(len(input_list)/2)
        random.shuffle(input_list)
        for i in range(num_of_pairs): # iterate over the pairs of nodes
            node1 = input_list[i*2]
            node2 = input_list[i*2+1]
            
            _, node1_out = adj_full_directed[node1, :].nonzero()
            _, node2_out = adj_full_directed[node2, :].nonzero()
            
            for n in node1_out: # exchange
                data_unstable[node1, n] = False
                data_unstable[n, node1] = False
                
                data_unstable[node2, n] = True
                data_unstable[n, node2] = True
            for n in node2_out: # exchange
                data_unstable[node2, n] = False
                data_unstable[n, node2] = False
                
                data_unstable[node1, n] = True
                data_unstable[n, node1] = True
    
    ############# stable nodes #############
    input_num_dict = {}   
    for i in stable_nodes_list:
        rows, _ = adj_full_directed[:, i].nonzero() # the column represent the nodes that points to i
        if i in rows: # skip self-loop/ PO
            continue
        num_of_inputs = len(rows)
        input_num_dict.setdefault(num_of_inputs, []).append(i)
        
    data_stable = adj_full.toarray()

    for input_num, input_list in input_num_dict.items(): # iterate over the number of inputs
        num_of_pairs = int(len(input_list)/2)
        random.shuffle(input_list)
        for i in range(num_of_pairs): # iterate over the pairs of nodes
            node1 = input_list[i*2]
            node2 = input_list[i*2+1]
            
            _, node1_out = adj_full_directed[node1, :].nonzero()
            _, node2_out = adj_full_directed[node2, :].nonzero()
            
            for n in node1_out: # exchange
                data_stable[node1, n] = False
                data_stable[n, node1] = False
                
                data_stable[node2, n] = True
                data_stable[n, node2] = True
            for n in node2_out: # exchange
                data_stable[node2, n] = False
                data_stable[n, node2] = False
                
                data_stable[node1, n] = True
                data_stable[n, node1] = True
    
    return csr_matrix(data_unstable), csr_matrix(data_stable)
    

def load_data(prefix, normalize=True, perturb_rate=0.1): 
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
    adj_full_directed = scipy.sparse.load_npz('./{}/adj_full_directed.npz'.format(prefix)).astype(bool)
    adj_test = scipy.sparse.load_npz('./{}/adj_test.npz'.format(prefix)).astype(bool)
    adj_full_unstable, adj_full_stable = apply_perturbation(adj_full, adj_full_directed, perturb_rate=perturb_rate)

    # adj_full_unstable, adj_full_stable = (0,0)
    print("nodes num = ", np.array(list(set(adj_test.nonzero()[0]))).shape[0])
    
    
    self_loops = adj_test.diagonal().nonzero()[0].size
    total_edges_including_duplicates = adj_test.nnz
    correct_number_of_edges = (total_edges_including_duplicates - self_loops) // 2 + self_loops
    print("Number of edges:", correct_number_of_edges)

    adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(prefix)).astype(bool)
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
    return adj_full, adj_full_unstable, adj_full_stable, adj_test, adj_train, feats, class_map, role

def process_graph_data(adj_full, adj_unstable, adj_stable, adj_test, adj_train, feats, class_map, role):
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
    return adj_full, adj_unstable, adj_stable, adj_test, adj_train, feats, class_arr, role

def parse_n_prepare(flags, perturb_rate=0.1):
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
    # print("Loading training data..")
    print("Prefix: ", flags.data_prefix)
    temp_data = load_data(flags.data_prefix, perturb_rate=perturb_rate)
    train_data = process_graph_data(*temp_data)
    # print("Done loading training data..")
    return train_params,train_phases,train_data,arch_gcn

def prepare(train_data,train_params,arch_gcn):
    """
    Prepare some data structure and initialize model / minibatch handler before
    the actual iterative training taking place.
    """
    adj_full, adj_unstable, adj_stable, adj_test, adj_train, feat_full, class_arr,role = train_data
    adj_full = adj_full.astype(np.int32)
    adj_unstable = adj_unstable.astype(np.int32)
    adj_stable = adj_stable.astype(np.int32)
    adj_test = adj_test.astype(np.int32)
    adj_train = adj_train.astype(np.int32)
    adj_full_norm = adj_norm(adj_full)
    adj_unstable_norm = adj_norm(adj_unstable)
    adj_stable_norm = adj_norm(adj_stable)
    num_classes = class_arr.shape[1]

    minibatch = Minibatch(adj_full_norm, adj_train, role, train_params)
    minibatch_unstable = Minibatch(adj_unstable_norm, adj_train, role, train_params)
    minibatch_stable = Minibatch(adj_stable_norm, adj_train, role, train_params)
    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    printf("TOTAL NUM OF PARAMS = {}".format(sum(p.numel() for p in model.parameters())))
    minibatch_eval=Minibatch(adj_full_norm, adj_train, role, train_params, cpu_eval=True)
    minibatch_unstable_eval=Minibatch(adj_unstable_norm, adj_train, role, train_params, cpu_eval=True)
    minibatch_stable_eval=Minibatch(adj_stable_norm, adj_train, role, train_params, cpu_eval=True)
    model_eval=GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    if args_global.gpu >= 0:
        model = model.cuda()
    return model, (minibatch, minibatch_unstable, minibatch_stable), (minibatch_eval, minibatch_unstable_eval, minibatch_stable_eval), model_eval, (adj_full, adj_unstable, adj_stable, adj_test)

def evaluate_full_batch(model, minibatch, mode='val'):
    """
    Full batch evaluation: for validation and test sets only.
        When calculating the F1 score, we will mask the relevant root nodes
        (e.g., those belonging to the val / test sets).
    """
    loss,preds,labels, embedding = model.eval_step(*minibatch.one_batch(mode=mode))
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
    # loss is not very accurate in this case, since loss is also contributed by training nodes
    # on the other hand, for val / test, we mostly care about their accuracy only.
    # so the loss issue is not a problem.
    return loss, f1mic, f1mac, embedding



def inference(model, minibatch_eval, model_eval, adj_test):
    path_saver = '{}/pytorch_models/saved_model_2024-04-08 23-39-59.pkl'.format(args_global.dir_log) # interconnected_modulus
    # path_saver = '{}/pytorch_models/saved_model_2024-04-27 17-55-09.pkl'.format(args_global.dir_log) # add_mul_mux
    # path_saver = '{}/pytorch_models/saved_model_{}.pkl'.format(args_global.dir_log, os.path.basename(args_global.data_prefix))
    model.load_state_dict(torch.load(path_saver))
    model_eval=model
    test_nodes = np.array(list(set(adj_test.nonzero()[0])))
    
    
    # embedding = model.inference_embedding(*minibatch.one_batch(mode='valtest'))
    loss, f1mic_both, f1mac_both, embedding = evaluate_full_batch(model_eval, minibatch_eval, mode='test')
    test_embedding = embedding[test_nodes]
    # f1mic_val, f1mic_test = f1mic_both
    # f1mac_val, f1mac_test = f1mac_both
    printf("Full validation: \n  F1_Micro = {:.4f}\tF1_Macro = {:.4f}"\
            .format(f1mic_both, f1mac_both))
    
    return test_embedding

if __name__ == '__main__':
    
    
    perturb_rate = [0.05, 0.1 , 0.2]
    
    for i in perturb_rate:
        print("Perturbation rate: %.2f" % i)
        train_params, train_phases, train_data, arch_gcn = parse_n_prepare(args_global, perturb_rate=i)
        model, minibatchs, minibatch_evals, model_eval, adj_totals = prepare(train_data, train_params, arch_gcn)
        minibatch, minibatch_unstable, minibatch_stable = minibatchs
        minibatch_eval, minibatch_unstable_eval, minibatch_stable_eval = minibatch_evals
        adj_full, adj_unstable, adj_stable, adj_test = adj_totals
        print("Before perturbation:")
        embedding = inference(model, minibatch, model_eval, adj_test)
        print("After perturbation unstable:")
        embedding_unstable = inference(model, minibatch_unstable, model_eval, adj_test)
        print("After perturbation stable:")
        embedding_stable = inference(model, minibatch_stable, model_eval, adj_test)
        
       
        test_nodes = np.array(list(set(adj_test.nonzero()[0])))
        assert test_nodes[0] == 0
        assert test_nodes[-1] == len(test_nodes) - 1
        endnodes = test_nodes[-1] + 1
        adj_test = adj_test[0:endnodes, 0:endnodes]
        # n_components, labels = connected_components(adj_test, directed=False, return_labels=True)
        # print("Number of connected components:", n_components)
        # print("Number of nodes in the largest connected component:", np.max(np.bincount(labels)))
        # print("Number of nodes in the smallest connected component:", np.min(np.bincount(labels)))
        embedding = embedding[:endnodes]
        embedding_unstable = embedding_unstable[:endnodes]
        embedding_stable = embedding_stable[:endnodes]
        
        
        cos_sim_unstable = cosine_similarity(embedding, embedding_unstable, dim=1)
        cos_sim_stable   = cosine_similarity(embedding, embedding_stable, dim=1)
        
        average_cos_sim_unstable = cos_sim_unstable.cpu().numpy().mean()
        average_cos_sim_stable = cos_sim_stable.cpu().numpy().mean()
        
        print("Average cosine similarity between original and unstable nodes: %.4f" % average_cos_sim_unstable)
        print("Average cosine similarity between original and stable nodes: %.4f" % average_cos_sim_stable)
        
        cos_sim_stable_cpu = cos_sim_stable.cpu().numpy()
        cos_sim_unstable_cpu = cos_sim_unstable.cpu().numpy()
        
        # length = len(cos_sim_unstable_cpu)
        # unstable_value = []
        # stable_value = []
        # for j in range(len(cos_sim_unstable_cpu)):
        #     if cos_sim_unstable_cpu[j] != 1:
        #         unstable_value.append(cos_sim_unstable_cpu[j])
                
        # for j in range(len(cos_sim_stable_cpu)):
        #     if cos_sim_stable_cpu[j] != 1:
        #         stable_value.append(cos_sim_stable_cpu[j])
        
        # print("Fraction of unstable nodes with cosine similarity less than 1: %.4f" % (len(unstable_value)/length))
        # print("Fraction of stable nodes with cosine similarity less than 1: %.4f" % (len(stable_value)/length))
                

        # import matplotlib.pyplot as plt
        # plt.hist(unstable_value, bins=100, alpha=0.5, label='unstable')
        # plt.hist(stable_value, bins=100, alpha=0.5, label='stable')
        # plt.legend(loc='upper left')
        # plt.title('Cosine Similarity Distribution')
        # # plt.show()
        # plt.savefig('Cosine_Similarity_Distribution_{}.png'.format(i))
        # plt.clf()
    

    