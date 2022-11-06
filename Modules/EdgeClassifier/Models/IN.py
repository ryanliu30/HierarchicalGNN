import sys

import torch.nn as nn
import torch
from torch_scatter import scatter_add
from torch.utils.checkpoint import checkpoint

from ..edge_classifier_base import EdgeClassifierBase

sys.path.append("../..")
from gnn_utils import InteractionGNNCell
from utils import make_mlp


class InteractionGNNBlock(nn.Module):

    """
    An interaction network class
    """

    def __init__(self, hparams, iterations, emb = True):
        super().__init__()
            
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """                 
        
        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"],
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_node_layer"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
            layer_norm=hparams["layernorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_encoder = make_mlp(
            2 * (hparams["spatial_channels"]),
            hparams["hidden"],
            hparams["latent"],
            hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["hidden_activation"],
            hidden_activation=hparams["hidden_activation"],
        )

        # Initialize GNN blocks
        if hparams["share_weight"]:
            cell = InteractionGNNCell(hparams)
            ignn_cells = [
                cell
                for _ in range(iterations)
            ]
        else:
            ignn_cells = [
                InteractionGNNCell(hparams)
                for _ in range(iterations)
            ]
        
        self.ignn_cells = nn.ModuleList(ignn_cells)
        
        # output layers
        if emb:
            self.output_layer = make_mlp(
                hparams["latent"],
                hparams["hidden"],
                hparams["emb_dim"],
                hparams["output_layers"],
                layer_norm=hparams["layernorm"],
                output_activation= None,
                hidden_activation=hparams["hidden_output_activation"],
            )
        
        self.emb = emb
        self.hparams = hparams
        
    def forward(self, x, graph):
        
        x.requires_grad = True
        
        nodes = checkpoint(self.node_encoder, x)
        edges = checkpoint(self.edge_encoder, torch.cat([x[graph[0]], x[graph[1]]], dim=1))
        
        for layer in self.ignn_cells:
            nodes, edges= layer(nodes, edges, graph)
        
        if self.emb:
            embeddings = self.output_layer(nodes)
            embeddings = nn.functional.normalize(embeddings) 
            return embeddings, nodes, edges
        else:
            return nodes, edges

class EC_InteractionGNN(EdgeClassifierBase):

    """
    An interaction network edge classifier class
    """

    def __init__(self, hparams):
        super().__init__(hparams) 
        
        self.ignn_block = InteractionGNNBlock(hparams, hparams["n_interaction_graph_iters"], emb = False)
        
        self.edge_classifier = make_mlp(
            2*hparams["latent"],
            hparams["hidden"],
            1,
            hparams["output_layers"],
            layer_norm=hparams["layernorm"],
            output_activation= None,
            hidden_activation=hparams["hidden_output_activation"],
        )
        
    def forward(self, x, graph):
        
        x.requires_grad = True
        
        directed_graph = torch.cat([graph, graph.flip(0)], dim = 1)
        
        nodes, edges = self.ignn_block(x, directed_graph)
        
        scores = self.edge_classifier(torch.cat([edges[:graph.shape[1]],edges[graph.shape[1]:]], dim = 1)).squeeze()
        scores = torch.sigmoid(scores)
        return scores
