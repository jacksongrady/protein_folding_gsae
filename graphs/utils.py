import os
import numpy as np
import pickle
import torch

total_graphs = []
times = []
arr = os.listdir(".")
for i, entry in enumerate(arr):

    split1, split2 = entry.split('.')
    if split2 == 'pkl' and split1 != 'total_graphs':

        with open(entry, "rb") as file:
            graphs = pickle.load(file)
            print(len(graphs), entry)
            for graph in graphs:
                if len(graph.name) > 1:
                    print('help')
                print(graph.name[0])
                a, b = graph.name[0].split('_')
                a, val = b.split('-')

                val = int(val) + int(a) * 10000
                times.append(val)
                total_graphs.append(graph)

print(len(times))
np.save('times.npy', times)
#print(len(total_graphs))
#np.save('total_graphs1.npy', total_graphs)

            #total_graphs.append(graphs)

# flat_list = []
# for sublist in total_graphs:
#     for item in sublist:
#         flat_list.append(item)
with open("total_graphs.pkl", 'wb') as out:
    pickle.dump(total_graphs, out)