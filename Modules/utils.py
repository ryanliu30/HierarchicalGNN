import os

import torch
import scipy as sp
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torch_geometric.data import Data
import frnn
import random

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset_paths(input_dir, datatype_names):
    """
    load event file paths. These file should be in PyG Data format
    """
    all_events = []
    for name in datatype_names:
        events = os.listdir(os.path.join(input_dir, name))
        events = sorted([os.path.join(input_dir, name, event) for event in events])
        all_events.extend(events)
    random.seed(42)
    random.shuffle(all_events)
    return all_events

class TrackMLDataset(Dataset):
    def __init__(self, dirs, hparams, stage = "train", device = "cpu"):
        super().__init__()
        self.dirs = dirs
        self.num = len(dirs)
        self.device = device
        self.stage = stage
        self.hparams = hparams
        
    def __getitem__(self, key):
        """
        The event file should contain:
        event.x: 3D coords of hits
        event.cell_data: addtional data coming from detector, e.g. charge deposit
        event.pid: particle ID of the hit
        event.hid: hit id of the node
        event.pt: transverse momentum of the paritcle
        event.edge_index: constructed graph
        event.modulewise_true_edges: ground truth graph built by connecting adjacent hits from the same particle
        event.signal_true_edges: a subset of modulewise_true_edges where particles are reconstructable 
        event.y: modulewise ground truth label of event.edge_index
        event.y_pid: particle ground truth label of event.edge_index
        
        """
        
        # load the event        
        event = torch.load(self.dirs[key], map_location=torch.device(self.device))
        if "1GeV" in str(self.dirs[key]):
            event = Data.from_dict(event.__dict__) # handle older PyG data format
        
        # the MASK tensor filter out hits from event
        if self.hparams["noise"]:
            mask = (event.pid == event.pid) # If using noise then only filter out those with nan PID
        else:
            mask = (event.pid != 0) # If not using noise then filter out those with PID 0, which represent that they are noise
        if self.hparams["hard_ptcut"] > 0:
            mask = mask & (event.pt > self.hparams["hard_ptcut"]) # Hard background cut in pT
        if self.hparams["remove_isolated"]:
            node_mask = torch.zeros(event.pid.shape).bool()
            node_mask[event.edge_index.unique()] = torch.ones(1).bool() # Keep only those nodes with edges attached to it
            mask = mask & node_mask
        
        # Set the pT of noise hits to be 0
        event.pt[event.pid == 0] = 0
        
        # Provide inverse mask to invert the change when necessary (e.g. track evaluation with not modified files)
        inverse_mask = torch.zeros(len(event.pid)).long()
        inverse_mask[mask] = torch.arange(mask.sum())
        event.inverse_mask = torch.arange(len(mask))[mask]
        
        # Compute number of hits (nhits) of each particle
        _, inverse, counts = event.pid.unique(return_inverse = True, return_counts = True)
        event.nhits = counts[inverse]
        
        if self.hparams["primary"]:
            event.signal_mask = ((event.nhits >= self.hparams["n_hits"]) & (event.primary == 1))
        else:
            event.signal_mask = (event.nhits >= self.hparams["n_hits"])
        
        # Randomly remove edges if needed
        if "edge_dropping_ratio" in self.hparams:  
            if self.hparams["edge_dropping_ratio"] != 0:
                edge_mask = (torch.rand(event.edge_index.shape[1]) >= self.hparams["edge_dropping_ratio"])
                event.edge_index = event.edge_index[:, edge_mask]
                event.y, event.y_pid = event.y[edge_mask], event.y_pid[edge_mask]
        
        for i in ["y", "y_pid"]:
            graph_mask = mask[event.edge_index].all(0)
            event[i] = event[i][graph_mask]

        for i in ["modulewise_true_edges", "signal_true_edges", "edge_index"]:
            event[i] = event[i][:, mask[event[i]].all(0)]
            event[i] = inverse_mask[event[i]]

        for i in ["x", "cell_data", "pid", "hid", "pt", "signal_mask"]:
            event[i] = event[i][mask]
            
        if self.hparams["primary"]:
            event.primary = event.primary[mask]
            
        event.dir = self.dirs[key]
            
        return event
    
    def __len__(self):
        return self.num

###

def graph_intersection(
    pred_graph, truth_graph, using_weights=False, weights_bidir=None
):
    """
    calculate the intersection of two graphs
    """
    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)
    del e_1
    del e_2

    if using_weights:
        weights_list = weights_bidir.cpu().numpy()
        weights_sparse = sp.sparse.coo_matrix(
            (weights_list, l2), shape=(array_size, array_size)
        ).tocsr()
        del weights_list
        del l2
        new_weights = weights_sparse[e_intersection.astype("bool")]
        del weights_sparse
        new_weights = torch.from_numpy(np.array(new_weights)[0])

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.from_numpy(
        np.vstack([e_intersection.row, e_intersection.col])
    ).long()  # .to(device)
    y = torch.from_numpy(e_intersection.data > 0)  # .to(device)
    del e_intersection

    if using_weights:
        return new_pred_graph, y, new_weights
    else:
        return new_pred_graph, y

    
def make_mlp(
    input_size,
    hidden_size,
    output_size,
    hidden_layers,
    hidden_activation="GELU",
    output_activation="GELU",
    layer_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    sizes = [input_size] + [hidden_size]*(hidden_layers-1) + [output_size]
    # Hidden layers
    for i in range(hidden_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)
    
def find_neighbors(embedding1, embedding2, r_max=1.0, k_max=10):
    embedding1 = embedding1.clone().detach().reshape((1, embedding1.shape[0], embedding1.shape[1]))
    embedding2 = embedding2.clone().detach().reshape((1, embedding2.shape[0], embedding2.shape[1]))
    
    dists, idxs, _, _ = frnn.frnn_grid_points(points1 = embedding1,
                                          points2 = embedding2,
                                          lengths1 = None,
                                          lengths2 = None,
                                          K = k_max,
                                          r = r_max,
                                         )
    return idxs.squeeze(0)

def FRNN_graph(embeddings, r, k):
    """
    build fixed radius KNN graph with embeddings
    """
    idxs = find_neighbors(embeddings, embeddings, r_max=r, k_max=k)

    positive_idxs = (idxs.squeeze() >= 0)
    ind = torch.arange(idxs.shape[0], device = positive_idxs.device).unsqueeze(1).expand(idxs.shape)
    edges = torch.stack([ind[positive_idxs],
                        idxs[positive_idxs]
                        ], dim = 0)     
    return edges