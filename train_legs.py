# TSNet and Scatter (with Optional Edge Feature Support)
import numpy as np
import torch
from torch.nn import Linear
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add


# TODO (alex) this is pretty inefficient using a for loop, is there a faster way to do this?
# TODO (alex) only scatter higher diffusions using masking
def scatter_moments(graph, batch_indices, moments_returned=4):
    """ Compute specified statistical coefficients for each feature of each graph passed. The graphs expected are disjoint subgraphs within a single graph, whose feature tensor is passed as argument "graph."
        "batch_indices" connects each feature tensor to its home graph.
        "Moments_returned" specifies the number of statistical measurements to compute. If 1, only the mean is returned. If 2, the mean and variance. If 3, the mean, variance, and skew. If 4, the mean, variance, skew, and kurtosis.
        The output is a dictionary. You can obtain the mean by calling output["mean"] or output["skew"], etc."""
    # Step 1: Aggregate the features of each mini-batch graph into its own tensor
    graph_features = [torch.zeros(0) for i in range(torch.max(batch_indices) + 1)]
    for i, node_features in enumerate(
        graph
    ):  # Sort the graph features by graph, according to batch_indices. For each graph, create a tensor whose first row is the first element of each feature, etc.
        #        print("node features are",node_features)
        if (
            len(graph_features[batch_indices[i]]) == 0
        ):  # If this is the first feature added to this graph, fill it in with the features.
            graph_features[batch_indices[i]] = node_features.view(
                -1, 1, 1
            )  # .view(-1,1,1) changes [1,2,3] to [[1],[2],[3]],so that we can add each column to the respective row.
        else:
            graph_features[batch_indices[i]] = torch.cat(
                (graph_features[batch_indices[i]], node_features.view(-1, 1, 1)), dim=1
            )  # concatenates along columns

    statistical_moments = {"mean": torch.zeros(0)}
    if moments_returned >= 2:
        statistical_moments["variance"] = torch.zeros(0)
    if moments_returned >= 3:
        statistical_moments["skew"] = torch.zeros(0)
    if moments_returned >= 4:
        statistical_moments["kurtosis"] = torch.zeros(0)

    for data in graph_features:
        data = data.squeeze()
        def m(i):  # ith moment, computed with derivation data
            return torch.sum(deviation_data ** i, axis=1) / torch.sum(
                torch.ones(data.shape), axis=1
            )

        mean = torch.sum(data, axis=1) / torch.sum(torch.ones(data.shape), axis=1)
        if moments_returned >= 1:
            statistical_moments["mean"] = torch.cat(
                (statistical_moments["mean"], mean[None, ...]), dim=0
            )

        # produce matrix whose every row is data row - mean of data row
        tuple_collect = []
        for a in mean:
            mean_row = torch.ones(data.shape[1]) * a
            tuple_collect.append(
                mean_row[None, ...]
            )  # added dimension to concatenate with differentiation of rows
        # each row contains the deviation of the elements from the mean of the row
        deviation_data = data - torch.cat(tuple_collect, axis=0)

        # variance: difference of u and u mean, squared element wise, summed and divided by n-1
        variance = m(2)
        if moments_returned >= 2:
            statistical_moments["variance"] = torch.cat(
                (statistical_moments["variance"], variance[None, ...]), dim=0
            )

        # skew: 3rd moment divided by cubed standard deviation (sd = sqrt variance), with correction for division by zero (inf -> 0)
        skew = m(3) / (variance ** (3 / 2))
        skew[
            skew > 1000000000000000
        ] = 0  # multivalued tensor division by zero produces inf
        skew[
            skew != skew
        ] = 0  # single valued division by 0 produces nan. In both cases we replace with 0.
        if moments_returned >= 3:
            statistical_moments["skew"] = torch.cat(
                (statistical_moments["skew"], skew[None, ...]), dim=0
            )

        # kurtosis: fourth moment, divided by variance squared. Using Fischer's definition to subtract 3 (default in scipy)
        kurtosis = m(4) / (variance ** 2) - 3
        kurtosis[kurtosis > 1000000000000000] = -3
        kurtosis[kurtosis != kurtosis] = -3
        if moments_returned >= 4:
            statistical_moments["kurtosis"] = torch.cat(
                (statistical_moments["kurtosis"], kurtosis[None, ...]), dim=0
            )


    # Concatenate into one tensor (alex)
    statistical_moments = torch.cat([v for k,v in statistical_moments.items()], axis=1)
    return statistical_moments


class LazyLayer(torch.nn.Module):
    """ Currently a single elementwise multiplication with one laziness parameter per
    channel. this is run through a softmax so that this is a real laziness parameter
    """

    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(2, n))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = torch.nn.functional.softmax(self.weights, dim=0)
        return torch.sum(inp * s_weights, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)
        
def gcn_norm(edge_index, edge_weight=None, num_nodes=None,
             add_self_loops=False, dtype=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight


class Diffuse(MessagePassing):
    """ Implements low pass walk with optional weights
    """

    def __init__(
        self, in_channels, out_channels, trainable_laziness=False, fixed_weights=True
    ):
        super().__init__(aggr="add", node_dim=-3)  # "Add" aggregation.
        assert in_channels == out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        if not self.fixed_weights:
            self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        # turn off this step for simplicity
        if not self.fixed_weights:
            x = self.lin(x)

        # Step 3: Compute normalization
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), dtype=x.dtype)

        # Step 4-6: Start propagating messages.
        propogated = self.propagate(
            edge_index, edge_weight=edge_weight, size=None, x=x,
        )
        if not self.trainable_laziness:
            return 0.5 * (x + propogated)
        return self.lazy_layer(x, propogated)

    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return edge_weight.view(-1, 1, 1) * x_j
    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out


def feng_filters():
    tmp = np.arange(16).reshape(4,4) #tmp doesn't seem to be used!
    results = [4]
    for i in range(2, 4):
        for j in range(0, i):
            results.append(4*i+j)
    return results


class Scatter(torch.nn.Module):
    def __init__(self, in_channels, edge_in_channels = None, trainable_laziness=False):
        super().__init__()
        self.in_channels = in_channels
        self.edge_in_channels = edge_in_channels
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer1 = Diffuse(in_channels, in_channels, trainable_laziness=trainable_laziness)
        self.diffusion_layer2 = Diffuse(
            4 * in_channels, 4 * in_channels, trainable_laziness=trainable_laziness
        ) # No edge channels the second time.

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print(f"SHall we use edge features? The user specified {self.edge_in_channels}. Does the data have edge attributes?{hasattr(data,'edge_attr')}")
        if hasattr(data,'edge_attr') and self.edge_in_channels: # if you've specified edge channels and the data has them, we'll use them!
          #print("We're using edge attributes!")
          edge_attr = data.edge_attr
        else:
          edge_attr = None;
    
        avgs = [x[:,:,None]]
        for i in range(16):
            avgs.append(self.diffusion_layer1(avgs[-1], edge_index)) # when diffusing over the graph, we use edge features
        filter1 = avgs[1] - avgs[2]
        filter2 = avgs[2] - avgs[4]
        filter3 = avgs[4] - avgs[8]
        filter4 = avgs[8] - avgs[16]
        s0 = avgs[0]
        s1 = torch.abs(torch.cat([filter1, filter2, filter3, filter4], dim=-1))

        avgs = [s1]
        for i in range(16):
            avgs.append(self.diffusion_layer2(avgs[-1], edge_index)) # When diffusing over our filters, we don't/
        filter1 = avgs[1] - avgs[2]
        filter2 = avgs[2] - avgs[4]
        filter3 = avgs[4] - avgs[8]
        filter4 = avgs[8] - avgs[16]
        s2 = torch.abs(torch.cat([filter1, filter2, filter3, filter4], dim=1))
        s2_reshaped = torch.reshape(s2, (-1, self.in_channels, 4))
        s2_swapped = torch.reshape(torch.transpose(s2_reshaped, 1, 2), (-1, 16, self.in_channels))
        s2 = s2_swapped[:, feng_filters()]

        x = torch.cat([s0, s1], dim=2)
        x = torch.transpose(x, 1, 2)
        x = torch.cat([x, s2], dim=1)

        # x = scatter_mean(x, batch, dim=0)
        if hasattr(data, 'batch'):
            x = scatter_moments(x, data.batch, 4)
        else:
            x = scatter_moments(x, torch.zeros(data.x.shape[0], dtype=torch.int32), 4)
        return x

    def out_shape(self):
        # x * 4 moments * in
        return 11 * 4 * self.in_channels


class TSNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_in_channels = None, trainable_laziness=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels
        self.trainable_laziness = trainable_laziness
        self.scatter = Scatter(in_channels, edge_in_channels = edge_in_channels, trainable_laziness=trainable_laziness)
        self.lin1 = Linear(self.scatter.out_shape(), out_channels)
        self.act = torch.nn.LeakyReLU()

    def forward(self, data):
        x = self.scatter(data)
        x = self.act(x)
        x = self.lin1(x)
        return x

import json
import numpy as np
import networkx as nx
import torch
import torch.utils
from torch.nn import Linear
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.datasets import TUDataset
from tqdm import trange
from de_shaw_Dataset import DEShaw
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class NetworkXTransform(object):
    def __init__(self, cat=False):
        self.cat = cat

    def __call__(self, data):
        x = data.x
        netx_data = to_networkx(data)
        ecc = self.nx_transform(netx_data)
        nx.set_node_attributes(netx_data, ecc, 'x')
        ret_data = from_networkx(netx_data)
        ret_x = ret_data.x.view(-1, 1).type(torch.float32)
        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, ret_x], dim=-1)
        else:
            data.x = ret_x
        return data

    def nx_transform(self, networkx_data):
        """ returns a node dictionary with a single attribute
        """
        raise NotImplementedError


class Eccentricity(NetworkXTransform):
    def nx_transform(self, data):
        return nx.eccentricity(data)


class ClusteringCoefficient(NetworkXTransform):
    def nx_transform(self, data):
        return nx.clustering(data)


def get_transform(name):
    if name == "eccentricity":
        transform = Eccentricity()
    elif name == "clustering_coefficient":
        transform = ClusteringCoefficient()
    elif name == "scatter":
        transform = Compose([Eccentricity(), ClusteringCoefficient(cat=True)])
    else:
        raise NotImplementedError("Unknown transform %s" % name)
    return transform


def split_dataset(dataset, splits=(0.8, 0.1, 0.1), seed=0):
    """ Splits data into non-overlapping datasets of given proportions.
    """
    splits = np.array(splits)
    splits = splits / np.sum(splits)
    n = len(dataset)
    torch.manual_seed(seed)
    val_size = int(splits[1] * n)
    test_size = int(splits[2] * n)
    train_size = n - val_size - test_size
    #ds = dataset.shuffle()
    ds = dataset
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    #return ds[:train_size], ds[train_size : train_size + val_size], ds[-test_size:]
    return train_set, val_set, test_set


def accuracy(model, dataset,loss_fn, name):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    total_loss = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        total_loss += loss_fn(pred,data.y)
    # correct = float(pred.eq(data.y).sum().item())
    acc = total_loss / len(dataset)
    return acc, pred

class EarlyStopping(object):
    """ Early Stopping pytorch implementation from Stefano Nardo https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d """
    def __init__(self, mode='min', min_delta=0, patience=8, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if metrics != metrics: # slight modification from source, to handle non-tensor metrics. If NAN, return True.
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def evaluate(model,loss_fn, train_ds, test_ds, val_ds):
    train_acc, train_pred = accuracy(model, train_ds,loss_fn, "Train")
    test_acc, test_pred = accuracy(model, test_ds,loss_fn, "Test")
    val_acc, val_pred = accuracy(model, val_ds,loss_fn, "Test")
    results = {
        "train_acc": train_acc,
        "train_pred": train_pred,
        "test_acc": test_acc,
        "test_pred": test_pred,
        "val_acc": val_acc,
        "val_pred": val_pred,
        "state_dict": model.state_dict(),
    }
    return results

def train_model(run_args, out_file):

    if "transform" in run_args:
        transform = get_transform(run_args["transform"])
    else:
        transform = None
    print(f"Using Transform {transform}")

    if run_args["dataset"] in ["COLLAB", "REDDIT-MULTI-5K", "IMDB-BINARY","IMDB-MULTI","BZR","OHSU","QM9"]:
#         dataset = TUDataset(
#             root="", name=run_args["dataset"], pre_transform=transform, 
#             use_node_attr=True, use_edge_attr=True
#         )
        
#         return dataset
        dataset = DEShaw("graphs/total_graphs.pkl")
        train_ds, val_ds, test_ds = split_dataset(dataset,splits=args["splits"])
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8)

    elif run_args["dataset"] in ["ogbg-molhiv"]:
        from ogb.graphproppred import PygGraphPropPredDataset
        d_name = "ogbg-molhiv"
        dataset = PygGraphPropPredDataset(name=run_args["dataset"])

        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=8)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

    if run_args["model"] == "ts_net":
        model = TSNet(
            dataset.num_node_features,
            dataset.num_classes,
            #edge_in_channels=dataset.num_edge_features,
            trainable_laziness=False
        )
    else:
        raise NotImplementedError()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    early_stopper = EarlyStopping(mode = 'max',patience=5,percentage=True)

    results_compiled = []
    early_stopper = EarlyStopping(mode = 'min',patience=5,percentage=False)

    model.train()
    for epoch in trange(1, 300 + 1):
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            results = evaluate(model, loss_fn, train_ds, test_ds, val_ds)
            print('Epoch:', epoch, results['train_acc'], results['test_acc'])
            results_compiled.append(results['test_acc'])
            #torch.save(results, '%s_%d.%s' % (out_file, epoch, out_end))
            if early_stopper.step(results['val_acc']):
                print("Early stopping criterion met. Ending training.")
                break # if the validation accuracy decreases for eight consecutive epochs, break. 
    model.eval()
    results = evaluate(model, loss_fn, train_ds, test_ds, val_ds)
    print("Results compiled:",results_compiled)
    print('saving scatter model')
    torch.save(model.scatter.state_dict(), str(out_file) + "LEGS_module_deshaw.npy")
    torch.save(results, str(out_file) + "LEGS_results_deshaw.pth")

args = {
    "dataset": "QM9",
    "model": "ts_net",
    "model_args": {
        "epsilon": 1e-16,
        "num_layers": 1
    },
    "model_dir": "/home/atong/trainable_scattering/models/v1/0",
    "transform": "clustering_coefficient", # QM9 contains "infinite path lengths," which the eccentricity
    "splits":(0.8,0.1,0.1)
}


dataset = train_model(args, './results/')