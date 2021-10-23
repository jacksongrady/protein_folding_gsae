# protein_folding_gsae


All you have to do is the following to train GSAE with the 'LEGS' version of scattering implemented (only works on orkney):

0.5) create a directory called 'graphs'
1) run 'get_data.py' to get graph data for the GB3 protein (the specific protein can be changed in the file!)
2) run 'train_gsae.py'
3) run 'utils.py' with 'model.npy' in current directory to get embeddings, a list of times, and the list of scattering moments
for a specific trained gsae model
