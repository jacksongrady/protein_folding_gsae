import pickle
from de_shaw_Dataset import DEShaw, Scattering
import torch

with open('graphs/total_graphs.pkl', 'rb') as file:
            graphs = pickle.load(file)


for i, entry in enumerate(graphs):
    if (type(entry)==list):
        print('alert', i)
        break
# with open('graphs/total_graphs.pkl', 'rb') as file:
#             graphs = pickle.load(file)

# print(graphs[0].node_id)


#MET, GLN, TYR, LYS, LEU, VAL, ILE, ASN, GLY, THR, GLU, ALA, ASP, PHE, TRP