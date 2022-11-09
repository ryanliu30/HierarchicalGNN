import sys

import torch.nn as nn
import torch
from torch_scatter import scatter_add
from torch.utils.checkpoint import checkpoint
from cugraph.structure.symmetrize import symmetrize
import cudf
import cupy as cp

sys.path.append("..")
from utils import make_mlp, find_neighbors

def checkpointing(func):
    return lambda *x: checkpoint(func, *x)

class InteractionGNNCell(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        # The edge network computes new edge features
        self.edge_network = make_mlp(
            3 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            2 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.hparams = hparams
    
    @checkpointing
    def node_update(self, nodes, edges, graph):
        """
        Calculate node update with checkpointing
        """
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        node_input = torch.cat([nodes, edge_messages], dim=-1)
        nodes = self.node_network(node_input) + nodes # Skip connection
        
        return nodes
    
    @checkpointing
    def edge_update(self, nodes, edges, graph):
        """
        Calculate edge update with checkpointing
        """
        edge_input = torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)
        edges = self.edge_network(edge_input) + edges # Skip connection
        
        return edges        
    
    def forward(self, nodes, edges, graph):

        nodes = self.node_update(nodes, edges, graph)
        edges = self.edge_update(nodes, edges, graph)

        return nodes, edges

class HierarchicalGNNCell(nn.Module):
    def __init__(self, hparams):
        super().__init__()  
        
        self.edge_network = make_mlp(
            3 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )

        self.node_network = make_mlp(
            3 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.supernode_network = make_mlp(
            3 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.superedge_network = make_mlp(
            3 * hparams["latent"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation="Tanh",
            hidden_activation=hparams["hidden_activation"],
        )
        
        self.hparams = hparams
    
    @checkpointing 
    def node_update(self, nodes, edges, supernodes, graph, bipartite_graph, bipartite_edge_weights):
        """
        Calculate node updates with checkpointing
        """
        supernode_messages = scatter_add(bipartite_edge_weights*supernodes[bipartite_graph[1]], bipartite_graph[0], dim=0, dim_size=nodes.shape[0])
        edge_messages = scatter_add(edges, graph[1], dim=0, dim_size=nodes.shape[0])
        nodes = self.node_network(torch.cat([nodes, edge_messages, supernode_messages], dim=-1)) + nodes
        return nodes
    
    @checkpointing 
    def edge_update(self, nodes, edges, graph):
        """
        Calculate edge updates with checkpointing
        """
        edges = self.edge_network(torch.cat([nodes[graph[0]], nodes[graph[1]], edges], dim=-1)) + edges
        return edges
    
    @checkpointing 
    def supernode_update(self, nodes, supernodes, superedges, bipartite_graph, bipartite_edge_weights, super_graph, super_edge_weights):
        """
        Calculate supernode updates with checkpointing
        """
        node_messages = scatter_add(bipartite_edge_weights*nodes[bipartite_graph[0]], bipartite_graph[1], dim=0, dim_size=supernodes.shape[0])
        attention_messages = scatter_add(superedges*super_edge_weights, super_graph[1], dim=0, dim_size=supernodes.shape[0])
        supernodes = self.supernode_network(torch.cat([supernodes, attention_messages, node_messages], dim=-1)) + supernodes
        return supernodes
    
    @checkpointing
    def superedge_update(self, supernodes, superedges, super_graph, super_edge_weights):
        """
        Calculate superedge updates with checkpointing
        """
        superedges = self.superedge_network(torch.cat([supernodes[super_graph[0]], supernodes[super_graph[1]], superedges], dim=-1)) + superedges
        return superedges
       
    def forward(self, nodes, edges, supernodes, superedges, graph, bipartite_graph, bipartite_edge_weights, super_graph, super_edge_weights):
        """
        Whereas the message passing in the original/super graphs is implemented by interaction network, the one in between them (bipartite message 
        passing is descirbed by weighted graph convolution (vanilla aggregation without attention)
        """
        
        # Compute new node features
        supernodes = self.supernode_update(nodes, supernodes, superedges, bipartite_graph, bipartite_edge_weights, super_graph, super_edge_weights)
        nodes = self.node_update(nodes, edges, supernodes, graph, bipartite_graph, bipartite_edge_weights)
        
        # Compute new edge features
        superedges = self.superedge_update(supernodes, superedges, super_graph, super_edge_weights)
        edges = self.edge_update(nodes, edges, graph)
        
        return nodes, edges, supernodes, superedges

class DynamicGraphConstruction(nn.Module):
    def __init__(self, weighting_function, hparams):
        """
        weighting function is used to turn dot products into weights
        """
        super().__init__()
        
        self.hparams = hparams
        self.weight_normalization = nn.BatchNorm1d(1)  
        self.weighting_function = getattr(torch, weighting_function)
        self.register_buffer("knn_radius", torch.ones(1), persistent=True)
        
    def forward(self, src_embeddings, dst_embeddings, sym = False, norm = False, k = 10, logits = False):
        """
        src embeddings: source nodes' embeddings
        dst embeddings: destination nodes' embeddings
        sym: whether to symmetrize the graph or not
        norm: whether to normalize the sum of weights around each not to 1 or not; empirically using True gives better convergence
        k: the source degree of the output graph
        logits: whether to output logits (dot products) or not
        """
        # Construct the Graph
        with torch.no_grad():            
            graph_idxs = find_neighbors(src_embeddings, dst_embeddings, r_max=self.knn_radius, k_max=k)
            positive_idxs = (graph_idxs >= 0)
            ind = torch.arange(graph_idxs.shape[0], device = src_embeddings.device).unsqueeze(1).expand(graph_idxs.shape)
            if sym:
                src, dst = symmetrize(cudf.Series(ind[positive_idxs]), cudf.Series(graph_idxs[positive_idxs]))
                graph = torch.tensor(cp.vstack([src.to_cupy(), dst.to_cupy()]), device=src_embeddings.device).long()
            else:
                src, dst = ind[positive_idxs], graph_idxs[positive_idxs]
                graph = torch.stack([src, dst], dim = 0)
            if self.training:
                maximum_dist = (src_embeddings[graph[0]] - dst_embeddings[graph[1]]).square().sum(-1).sqrt().max()
                self.knn_radius = 0.9*self.knn_radius + 0.11*maximum_dist # Keep track of the minimum radius needed to give right number of neighbors
        
        # Compute bipartite attention
        likelihood = torch.einsum('ij,ij->i', src_embeddings[graph[0]], dst_embeddings[graph[1]]) 
        edge_weights_logits = self.weight_normalization(likelihood.unsqueeze(1)).squeeze() # regularize to ensure variance of weights
        edge_weights = self.weighting_function(edge_weights_logits)
        
        if norm:
            edge_weights = edge_weights/edge_weights.mean()
        edge_weights = edge_weights.unsqueeze(1)
        if logits:
            return graph, edge_weights, edge_weights_logits

        return graph, edge_weights
    
