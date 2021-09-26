import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm.notebook import tqdm
import networkx as nx
#from torch_geometric.data import Data
#from torch_geometric.utils import from_networkx
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score
import pickle
import warnings


#warnings.filterwarnings("ignore")
pdbs = np.load("file_names/pdgs0to2.npy")
# df = pd.read_csv("../datasets/pscdb/structural_rearrangement_data.csv")

# pdbs = df["Free PDB"]
#y = [torch.argmax(torch.Tensor(lab)).type(torch.LongTensor) for lab in LabelBinarizer().fit_transform(df.motion_type)]

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_peptide_bonds, add_k_nn_edges
from graphein.protein.graphs import construct_graph

from functools import partial

# Override config with constructors
constructors = {
    "edge_construction_functions": [partial(add_k_nn_edges, k=3, long_interaction_threshold=0)],
    "pdb_dir": "/data/lab/de_shaw/all_trajectory_slices/GB3/0 to 2 us",
    #"edge_construction_functions": [add_hydrogen_bond_interactions, add_peptide_bonds],
    #"node_metadata_functions": [add_dssp_feature]
}

config = ProteinGraphConfig(**constructors)
print(config.dict())

# Make graphs
graph_list = []
y_list = []
print(pdbs)
for idx, pdb in enumerate(tqdm(pdbs)):
    path = "/data/lab/de_shaw/all_trajectory_slices/GB3/0 to 2 us" + '/' + pdb
    try:
        graph_list.append(
            construct_graph(pdb_path=path,
                        config=config
                       )
            )
        #y_list.append(y[idx])
    except:
        print(str(idx) + ' processing error...')
        break
        pass

from graphein.ml.conversion import GraphFormatConvertor

format_convertor = GraphFormatConvertor('nx', 'pyg',
                                        verbose = 'gnn',
                                        columns = None)

pyg_list = [format_convertor(graph) for graph in tqdm(graph_list)]
# for i in pyg_list:
#     if i.coords.shape[0] == len(i.node_id):
#         pass
#     else:
#         print(i)
#         pyg_list.remove(i)
with open("graphs/0to2graphs.pkl", "wb") as file:
    pickle.dump(pyg_list, file )
