from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch_geometric.data
import pickle
from LEGS_module import Scatter

#from torch_geometric.utils.convert import from_networkx

import networkx as nx

class DEShaw(Dataset):
    """ZINCTranch dataset."""

    def __init__(self, file_name, transform=None):
        with open(file_name, 'rb') as file:
            self.graphs = pickle.load(file)
        
        self.amino_acid_dict = {'MET' : 0,\
                                'GLN' : 1,\
                                'TYR' : 2,\
                                'LYS' : 3,\
                                'LEU' : 4,\
                                'VAL' : 5,\
                                'ILE' : 6,\
                                'ASN' : 7,\
                                'GLY' : 8,\
                                'THR' : 9,\
                                'GLU' : 10,\
                                'ALA' : 11,\
                                'ASP' : 12,\
                                'PHE' : 13,\
                                'TRP' : 14,\

        }
        self.num_node_features = len(self.amino_acid_dict.keys())
        self.transform = transform

    def __len__(self):
        
        return len(self.graphs)

    def __getitem__(self, idx):
             
        data = self.graphs[idx]

       
        feats = []
        
        nodes = data.node_id


        for i in range(data.num_nodes):
            arr = np.zeros(15)
            acid = nodes[i]
            a, acid, index = acid.split(':')
            index = self.amino_acid_dict[acid]
            arr[index] = 1

            
#             arr[0] = 1.
#             arr[1] = 1. if data.element[i] == 'C' else 0.
#             arr[2] = 1. if data.element[i] == 'O' else 0.
#             arr[3] = 1. if data.element[i] == 'N' else 0.
            feats.append(arr)
        data.x = torch.tensor(feats).float()
        data.edge_attr = None

        if self.transform: 
            return self.transform(data)
        else:
            return data

class Scattering(object):

    def __init__(self):
        model = Scatter(15, trainable_laziness=None)
        #model.load_state_dict(torch.load("/home/jacksongrady/graphGeneration/gsae/gsae/LEGS/results/learnable_scat_model.npy"))
        model.eval()
        self.model = model
    
    def __call__(self, sample):
        #props = sample.y

        # elements = sample.element
        # carbon_arr =[]
        # oxy_arr = []
        # nitro_arr = []
        # for entry in elements:
        #     if entry == 'C':
        #         carbon_arr.append([1.])
        #     else:
        #         carbon_arr.append([0.]) 
        #     if entry == 'O':
        #         oxy_arr.append([1.])
        #     else:
        #         oxy_arr.append([0.])
        #     if entry == 'N':
        #         nitro_arr.append([1.])
        #     else:
        #         nitro_arr.append([0.])

        # carbon_arr = torch.Tensor(carbon_arr)
        # oxy_arr = torch.Tensor(oxy_arr)
        # nitro_arr = torch.Tensor(nitro_arr)

        # scat = self.model(sample)
        # sample.x = carbon_arr
        # carb = self.model(sample)
        # sample.x = oxy_arr
        # oxy = self.model(sample)
        # sample.x = nitro_arr
        # nitro = self.model(sample)
        with torch.no_grad():
            to_return = self.model(sample)
        #appended = torch.cat((scat[0][0].detach(), carb[0][0].detach(), oxy[0][0].detach(), nitro[0][0].detach()), 0)
        
        return to_return[0], 1
        