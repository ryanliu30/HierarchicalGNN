import torch
import cupy as cp
import numpy as np
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
from matplotlib import cm
from sklearn.manifold import TSNE
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

default_response = {
    "track_eff": 0,
    "track_pur": 0,
    "hit_eff": 0,
    "hit_pur": 0
}

def eval_metrics(bipartite_graph, event, pt_cut = 1., nhits_cut = 5, majority_cut = 0.5, primary = True):
    """
    Evaluate Tracking Performance
    bipartite_graph: a bipartite graph connecting hits to track candidate. (i, j) means hit i is assigned to track j
    event: event PyG file, should at least contain pid, pt and primary information
    pt_cut: reconstructable criterion
    nhits_cut: reconstructable criterion
    primary: reconstructable criterion
    majority_cut: matching criterion
    """
    # remove small track candidate that cannot pass hit efficiency filter for any reconstructable particle
    _, clusters, counts = bipartite_graph[1].unique(return_inverse = True, return_counts = True)
    bipartite_graph = bipartite_graph[:, counts[clusters] >= (nhits_cut * majority_cut)] 
    # Relabel track candidates in ascending order
    bipartite_graph[1] = bipartite_graph[1].unique(return_inverse = True)[1]
    # Relable particles in ascending index
    original_pid, pid, nhits = torch.unique(event.pid, return_inverse = True, return_counts = True)
    
    if primary and ("primary" in event):
        primary_mask = (scatter_sum(event.primary, pid) > 0)
        primary_mask = cp.array(primary_mask.cpu().numpy())
    
    # define particlewise pT and nhit
    pt = scatter_min(event.pt, pid, dim=0, dim_size = pid.max()+1)[0]
    bipartite_graph, original_pid, pid, pt, nhits = cp.asarray(bipartite_graph), cp.asarray(original_pid), cp.asarray(pid), cp.asarray(pt), cp.asarray(nhits)
    
    # Construct particle counts matrix in each cluster
    pid_cluster_mapping = cp.sparse.coo_matrix((cp.ones(bipartite_graph.shape[1]), (pid[bipartite_graph[0]], bipartite_graph[1])), shape=(pid.max().item()+1, bipartite_graph[1].max().item()+1)).tocsr()
    
    # Use cluster hashing to avoid matching one particle to multiple candidate
    cluster_hashing = cp.linspace(1, 1+1e-12, bipartite_graph[1].max().item()+1).reshape(1, -1)
    matching = (pid_cluster_mapping >= majority_cut*pid_cluster_mapping.sum(0)) & (pid_cluster_mapping >= majority_cut*nhits.reshape(-1, 1)) & (pid_cluster_mapping.multiply(cluster_hashing) == pid_cluster_mapping.multiply(cluster_hashing).max(1).todense())
    
    row_match, col_match = cp.where(matching)
    if row_match.shape[0] == 0:
        return default_response # no matching found
    
    # filter out candidate that was matched to noise
    matching_mask = ((pid_cluster_mapping[row_match, col_match] > majority_cut*nhits_cut)[0] & (original_pid[row_match] != 0))
    row_match, col_match = row_match[matching_mask], col_match[matching_mask]

    if row_match.shape[0] == 0:
        return default_response # no matching found
    
    mask = (pt[row_match] > pt_cut) & (nhits[row_match] >= nhits_cut) # Reconstructable mask
    truth_mask = (pt > pt_cut) & (nhits >= nhits_cut) # Reconstructable mask for truth
    selected_hits = (pt[pid] > pt_cut) & (original_pid[pid] != 0) & (nhits[pid] >= nhits_cut) # hit-wise reconstructable mask
    
    if primary:
        # primary filtering if using primary as well
        mask = mask & primary_mask[row_match]
        truth_mask = truth_mask & primary_mask
        selected_hits = selected_hits & primary_mask[pid]

    # Tracking metric
    track_eff = mask.sum()/truth_mask.sum()
    hit_pur = (pid_cluster_mapping[row_match, col_match]/pid_cluster_mapping[:, col_match].sum(0)).mean()
    track_pur = mask.sum()/(pid_cluster_mapping.shape[1] - (~matching_mask).sum() - (~mask).sum())    
    hit_eff = (pid_cluster_mapping[row_match, col_match][mask.reshape(1, -1)]/(nhits[row_match][mask])).mean()

    return {
        "track_eff": track_eff.item(),
        "track_pur": track_pur.item(),
        "hit_eff": hit_eff.item(),
        "hit_pur": hit_pur.item()
    }        
        
        
        
    