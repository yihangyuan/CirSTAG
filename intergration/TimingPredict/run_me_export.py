import torch
import dgl
import random
import time
import datetime
import torch
import numpy as np
import dgl
import torch.nn.functional as F
import random
import pdb
import time
import os
from sklearn.metrics import r2_score
import tee
import pickle
from data_graph import data_train, data_test
from model import TimingGCN
from my_utils.utils import spectral_embedding,spectral_embedding_eig,SAGMAN
import argparse
from scipy.sparse import coo_matrix, csr_matrix
import copy
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shutil


parser = argparse.ArgumentParser(description="to select scale parameter")
parser.add_argument('--s', type=float, default=10, help="perturbation scale")
parser.add_argument('--f', type=float, default=0.1, help="select top stable and unstable fraction from rank")
args = parser.parse_args()


def gen_topo(g_hetero):
    torch.cuda.synchronize()
    time_s = time.time()
    na, nb = g_hetero.edges(etype='net_out', form='uv')
    ca, cb = g_hetero.edges(etype='cell_out', form='uv')
    g = dgl.graph((torch.cat([na, ca]).cpu(), torch.cat([nb, cb]).cpu()))
    topo = dgl.topological_nodes_generator(g)
    ret = [t.cuda() for t in topo]
    torch.cuda.synchronize()
    time_e = time.time()
    return ret, time_e - time_s

def adj_dgl2scipy(sparse_matrix):
    # Call the indices as a function
    indices = sparse_matrix.indices().cpu().numpy()
    shape = sparse_matrix.shape
    # Assuming the non-zero values are 1
    values = np.ones(len(indices[0]), dtype=np.float32)
    # Converting indices to appropriate type
    rows = indices[0].astype('int64')
    cols = indices[1].astype('int64')
    # Creating a scipy coo_matrix
    from scipy.sparse import coo_matrix
    coo_mat = coo_matrix((values, (rows, cols)), shape=shape)

    return coo_mat.tocsr()

def is_symmetric(matrix, tol=1e-10):
    if not isinstance(matrix, csr_matrix):
        raise ValueError("Input must be a csr_matrix")
    
    # Ensure the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Compute the difference between the matrix and its transpose
    diff = matrix - matrix.transpose()
    
    # Check if all elements are close to zero within the tolerance
    return np.all(np.abs(diff.data) < tol)

def test_pertubation_pin_cap(dataset,loaded_SAGMAN_dict_NodeList, name, model, baseline, scal=0.1, f=0.05, drop=True):

    dataset_unstable = copy.deepcopy(dataset)

    dataset_stable = copy.deepcopy(dataset)


    g = dataset_unstable[name][0]
    ts = dataset_unstable[name][1]
    print("Design: " + name )
    length = len(loaded_SAGMAN_dict_NodeList[name]) # node number
    num = int(length * f)


    unstable = []
    stable = []
    count_unstable = -1
    count_stable = 0

    if drop == True:
        for i in range(num):

            while True:
                count_unstable += 1
                index = loaded_SAGMAN_dict_NodeList[name][count_unstable]
                # (index in ts['output_nodes']) or
                if  (index in ts['output_nodes']) or(index in ts['pi_nodes']) or (index in ts['po_nodes']):   
                    continue
                else:
                    break
                # break
            
            unstable.append(index)

            while True:
                count_stable -= 1
                index = loaded_SAGMAN_dict_NodeList[name][count_stable]
                # (index in ts['output_nodes']) or 
                if  (index in ts['output_nodes']) or(index in ts['pi_nodes']) or (index in ts['po_nodes']):
                    continue
                else:
                    break
                # break
            stable.append(index)

    else:
        for i in range(num):
            unstable.append(loaded_SAGMAN_dict_NodeList[name][i])
            stable.append(loaded_SAGMAN_dict_NodeList[name][length - i - 1])

    
    for node in unstable:
        g.ndata['nf'][node][6:10] *= scal 

    for node in stable:
        dataset_stable[name][0].ndata['nf'][node][6:10] *= scal 
        
    with torch.no_grad():
        pred = model(g, ts, groundtruth=False)
        # net_delays, cell_delays, atslew = pred[0][:], pred[1][:], pred[2][:,:4]
        atslew = pred[2][:,:4]


    with torch.no_grad():
        pred_stable = model(dataset_stable[name][0], dataset_stable[name][1], groundtruth=False)
        # nd_stable, cd_stable, as_stable = pred_stable[0][:], pred_stable[1][:], pred_stable[2][:,:4]
        atslew_stable = pred_stable[2][:,:4]


    # a = baseline[name][0] # net_delay
    # b = baseline[name][2]
    as_baseline = baseline[name][2]

    result_unstable = []
    result_stable = []

    for node in ts['po_nodes']:
        result_unstable.append( ( (atslew[node] - as_baseline[node]) / as_baseline[node] ).cpu())


    for node in dataset_stable[name][1]['po_nodes']:
        result_stable.append( ( (atslew_stable[node] - as_baseline[node]) / as_baseline[node] ).cpu())
    
    return result_unstable, result_stable


random.seed(8026728)
available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'.split()
train_data_keys = random.sample(available_data, 14)
data = {}
current_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for k in available_data:

    g = dgl.load_graphs('data/8_rat/{}.graph.bin'.format(k))[0][0].to('cuda')
    g.ndata['n_net_delays_log'] = torch.log(0.0001 + g.ndata['n_net_delays']) + 7.6
    invalid_nodes = torch.abs(g.ndata['n_ats']) > 1e20   # ignore all uninitialized stray pins
    g.ndata['n_ats'][invalid_nodes] = 0
    g.ndata['n_slews'][invalid_nodes] = 0
    g.ndata['n_atslew'] = torch.cat([
        g.ndata['n_ats'],
        torch.log(0.0001 + g.ndata['n_slews']) + 3
    ], dim=1)
    g.edges['cell_out'].data['ef'] = g.edges['cell_out'].data['ef'].type(torch.float32)
    g.edges['cell_out'].data['e_cell_delays'] = g.edges['cell_out'].data['e_cell_delays'].type(torch.float32)
    topo, topo_time = gen_topo(g)
    ts = {'input_nodes': (g.ndata['nf'][:, 1] < 0.5).nonzero().flatten().type(torch.int32),
          'output_nodes': (g.ndata['nf'][:, 1] > 0.5).nonzero().flatten().type(torch.int32),
          'output_nodes_nonpi': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] < 0.5).nonzero().flatten().type(torch.int32),
          'pi_nodes': torch.logical_and(g.ndata['nf'][:, 1] > 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
          'po_nodes': torch.logical_and(g.ndata['nf'][:, 1] < 0.5, g.ndata['nf'][:, 0] > 0.5).nonzero().flatten().type(torch.int32),
          'endpoints': (g.ndata['n_is_timing_endpt'] > 0.5).nonzero().flatten().type(torch.long),
          'topo': topo,
          'topo_time': topo_time}
    data[k] = g, ts

data_train = {k: t for k, t in data.items() if k in train_data_keys}
data_test = {k: t for k, t in data.items() if k not in train_data_keys}



model = TimingGCN()
model.cuda()
model.load_state_dict(torch.load('./checkpoints/{}/{}.pth'.format('08_atcd_specul', 15799)))


available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'.split()
select = 0
available_data = available_data[select].split()
k = available_data[0] 
print("Test on: " + available_data[0])
SAGMAN_dict_NodeList = {key: [] for key in available_data}

with torch.no_grad():

    if 1:
        g = data[k][0]
        ts = data[k][1]
        torch.cuda.synchronize()
    
        time_s = time.time()
        pred = model(g, ts, groundtruth=False)[2][:, :4]

        adj_cell_out = adj_dgl2scipy(g.adj(etype='cell_out'))
        adj_net_out = adj_dgl2scipy(g.adj(etype='net_out'))
        adj_net_in = adj_dgl2scipy(g.adj(etype='net_in'))
        adj_total = adj_net_in + adj_net_out + adj_cell_out + adj_cell_out.T
        print("is symmetric ",is_symmetric(adj_total))
        # embedd_in = spectral_embedding(adj_total,g.ndata['nf'].cpu().numpy().copy(),use_feature=True)
        embedd_in = spectral_embedding(adj_total,None,use_feature=False)
        start_time = time.time() 
        TopEig, TopEdgeList, TopNodeList, _ = SAGMAN(embedd_in, pred.cpu().numpy(), k=50, num_eigs=4,weighted=True,sparse=True)
        SAGMAN_dict_NodeList[k].append(TopNodeList)
        torch.cuda.synchronize()
        time_t = time.time()
        truth = g.ndata['n_atslew'][:, :4]
        r2 = r2_score(pred.cpu().numpy().reshape(-1),
                        truth.cpu().numpy().reshape(-1))
        print('{:15} r2 {:1.5f}, time {:2.5f}'.format(k, r2, time_t - time_s))

folder_name = './stability_ranks/{}/{}'.format(k, current_timestamp)
os.makedirs(folder_name)  
current_file = __file__
destination_path = os.path.join(folder_name, os.path.basename(current_file))
shutil.copy(current_file, destination_path)
os.chdir(folder_name)

save_name = 'SAGMAN_dict_NodeList.pkl'

with open(save_name, 'wb') as f:
    pickle.dump(SAGMAN_dict_NodeList, f)

for key, value in SAGMAN_dict_NodeList.items():
    if not isinstance(value[0], list):
        SAGMAN_dict_NodeList[key] = value[0].tolist() 
    else:
        SAGMAN_dict_NodeList[key] = value[0]
        


## calcualte baseline 
baseline = {}

g = data[k][0]
ts = data[k][1]
with torch.no_grad():
    pred = model(g, ts, groundtruth=False)
    net_delays, cell_delays, atslew = pred[0][:], pred[1][:], pred[2][:,:4]
    baseline[k] = (net_delays, cell_delays, atslew)
        

result_unstable, result_stable = test_pertubation_pin_cap(data, SAGMAN_dict_NodeList, k,model, baseline, scal=args.s, f=args.f)

filename = "./scal{}_f{}_cap_AS.csv".format(args.s, args.f)

with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)

    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        writer.writerow(["name", "node", "position", "relativeChange", "unstable"])

    for node in range(len(result_unstable)):

        nodeName = ts['po_nodes'][node].item()
        index = 0

        for cap in result_unstable[node]:
            writer.writerow([k, nodeName, index ,cap.item(), "True"])
            index += 1


    for node in range(len(result_stable)):

        nodeName = ts['po_nodes'][node].item()
        index = 0

        for cap in result_stable[node]:
            writer.writerow([k, nodeName, index, cap.item(), "False"])
            index += 1

save_data = data # temporary solution

data = pd.read_csv(filename)
data['relativeChange'] *= 100
data = data.rename(columns={'relativeChange': 'Relative Change %'})
data = data.rename(columns={'name': 'Benchmark'})
# rename hue
data['unstable'] = data['unstable'].map({True: 'Unstable', False: 'Stable'})
data = data.rename(columns={'unstable': 'Perturbation on'})
plt.figure(figsize=(3.5,2.75), 
           dpi = 400)
palette = None
hue_order = data['Perturbation on'].unique()
hue_order = hue_order[::-1]
ax = sns.stripplot(data=data, x="Relative Change %", y= "Benchmark", hue = "Perturbation on",palette = palette, jitter=0.5, size=1.5, dodge=True, hue_order=hue_order)
ax.legend(loc='lower right', fontsize='small')
ax.set_ylabel('')
plot_name = "./scal{}_f{}_cap_AS.png".format(args.s, args.f)
plt.savefig(plot_name, bbox_inches='tight')
# plt.show()
plt.close()


