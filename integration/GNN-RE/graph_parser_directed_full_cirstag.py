
import networkx as nx
import scipy.sparse as sp
import numpy as np


# os.chdir('./Netlist_to_graph/Graphs_datasets/Interconnected-Modules')




row=[]
col=[]
connect=[]
        
with open("row.txt", "r") as file_row:
    with open("col.txt", "r") as file_col:
        row_prev = -1
        col_prev = -1
        for row_line, col_line in zip(file_row, file_col):
            
            row_stripped = int(row_line.strip())
            col_stripped = int(col_line.strip())
            
            if row_stripped == col_prev and col_stripped == row_prev:
                continue
            else:
                row.append(row_stripped)
                col.append(col_stripped)
                connect.append(True)
                
                row_prev = row_stripped
                col_prev = col_stripped
        
        
        
row_ind = np.array(row)
col_ind = np.array(col)
data = np.array(connect, dtype=bool)
mat_coo = sp.coo_matrix((data, (row_ind, col_ind)))
sparse_matrix=mat_coo.tocsr()

sp.save_npz('adj_full_directed.npz', sparse_matrix)
