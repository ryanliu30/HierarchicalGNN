# System imports
import sys

# 3rd party imports
from pytorch_lightning import LightningModule
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import cupy as cp
from torch.utils.data import random_split
import torch.nn as nn
import cugraph
import cudf

device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append("../..")

# Local imports
from utils import load_dataset_paths, TrackMLDataset
from tracking_utils import eval_metrics

class EdgeClassifierBase(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different filter training regimes
        """
        self.save_hyperparameters(hparams)
        
    def setup(self, stage):
        
        paths = load_dataset_paths(self.hparams["input_dir"], self.hparams["datatype_names"])
        paths = paths[:sum(self.hparams["train_split"])]
        self.trainset, self.valset, self.testset = random_split(paths, self.hparams["train_split"], generator=torch.Generator().manual_seed(0))
        
    def train_dataloader(self):
        self.trainset = TrackMLDataset(self.trainset, self.hparams, stage = "train", device = "cpu")
        if self.trainset is not None:
            return DataLoader(self.trainset, batch_size=1, num_workers=16, shuffle = True)
        else:
            return None

    def val_dataloader(self):
        self.valset = TrackMLDataset(self.valset, self.hparams, stage = "val", device = "cpu")
        if self.valset is not None:
            return DataLoader(self.valset, batch_size=1, num_workers=16)
        else:
            return None

    def test_dataloader(self):
        self.testset = TrackMLDataset(self.testset, self.hparams, stage = "test", device = "cpu")
        if self.testset is not None:
            return DataLoader(self.testset, batch_size=1, num_workers=16)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler
    
    def pt_weighting(self, pt):
        """
        Take in pt and caculate weight w as following:
        w = weight_min + (1 - weight_min)*min(max(pt - pt_cut + pt_interval, 0), 1) + weight_leak*max(pt - ptcut, 0)
        """
        pt[pt!=pt] = 0

        h = lambda i: torch.heaviside(i, torch.zeros(1).to(pt))
        minimum = lambda i: torch.minimum(i, torch.ones(1).to(pt))
        
        eps = self.hparams["weight_leak"]
        cut = self.hparams["ptcut"] - self.hparams["pt_interval"]
        cap = self.hparams["ptcut"]
        min_weight = self.hparams["weight_min"]
        
        return min_weight + (1-min_weight)*minimum(h(pt-cut)*(pt-cut)/(cap-cut)) + (eps * h(pt-cap) * (pt-cap))
    
    def get_training_weight(self, batch, graph, y):
        """
        Calculate weights and balancing positive and negative samples
        Per edge weight is defined as the sum of pt weights of the two ends
        """
        weights = self.pt_weighting(batch.pt[graph[0]]) + self.pt_weighting(batch.pt[graph[1]])
        true_weights = weights[y].sum()
        fake_weights = weights[~y].sum()
        
        weights[y] = (weights[y]/true_weights)*torch.sigmoid(self.hparams["log_weight_ratio"]*torch.ones(1, device = self.device))
        weights[~y] = (weights[~y]/fake_weights)*torch.sigmoid(-self.hparams["log_weight_ratio"]*torch.ones(1, device = self.device))
        
        return weights.float()
    
    def training_step(self, batch, batch_idx):
        
        scores = self(batch.x, batch.edge_index)
        if self.hparams["true_edges"] == "modulewise_true_edges":
            # Neutral edges (PID true but not modulewise ones) are removed from training
            graph = batch.edge_index[:, (batch.y_pid == 0)|(batch.y == 1)]
            y = batch.y[(batch.y_pid == 0)|(batch.y == 1)]
            scores = scores[(batch.y_pid == 0)|(batch.y == 1)]
        elif self.hparams["true_edges"] == "pid_true_edges":
            graph = batch.edge_index
            y = batch.y_pid
            
        weights = self.get_training_weight(batch, graph, y.bool())
        
        loss = nn.functional.binary_cross_entropy(scores, y.float(), reduction='none')
        loss = torch.dot(loss, weights)
        
        self.log("training_loss", loss)
        
        return loss 


    def shared_evaluation(self, batch, batch_idx, log=False):

        """
        This method is shared between validation steps and test steps
        """
        
        scores = self(batch.x, batch.edge_index)
        
        if self.hparams["true_edges"] == "modulewise_true_edges":
            cut_graph = batch.edge_index[:, (batch.y_pid == 0)|(batch.y == 1)]
            cut_y = batch.y[(batch.y_pid == 0)|(batch.y == 1)]
            cut_scores = scores[(batch.y_pid == 0)|(batch.y == 1)]
        elif self.hparams["true_edges"] == "pid_true_edges":
            cut_graph = batch.edge_index
            cut_y = batch.y_pid
            cut_scores = scores
            
        weights = self.get_training_weight(batch, cut_graph, cut_y.bool())
        loss = nn.functional.binary_cross_entropy(cut_scores, cut_y.float(), reduction='none')
        loss = torch.dot(loss, weights)
        
        # Compute Tracking Efficiency        
        G = cugraph.Graph()
        df = cudf.DataFrame({"src": cp.asarray(batch.edge_index[0]),
                             "dst": cp.asarray(batch.edge_index[1]),
                             "weights": cp.asarray(scores)})
        if (scores >= self.hparams["score_cut"]).any(): # to ensure there's at least an edge
            df = df[df["weights"] >= self.hparams["score_cut"]]
        G.from_cudf_edgelist(df, source = "src", destination = "dst", edge_attr = "weights")
        connected_components = cugraph.components.connected_components(G)
        bipartite_graph = torch.stack([torch.as_tensor(connected_components["vertex"]), torch.as_tensor(connected_components["labels"])], dim = 0)
        
        # Evaluate using not modified event to avoid miscalculation
        bipartite_graph[0] = batch.inverse_mask[bipartite_graph[0]]
        event = torch.load(batch.dir[0], map_location=torch.device(self.device))
        if "1GeV" in str(batch.dir[0]):
            event = Data.from_dict(event.__dict__)
        event.pt[event.pid == 0] = 0
        _, inverse, counts = event.pid.unique(return_inverse = True, return_counts = True)
        event.nhits = counts[inverse]
        
        tracking_performace_metrics = eval_metrics(bipartite_graph,
                                                   event,
                                                   pt_cut = self.hparams["ptcut"],
                                                   nhits_cut = self.hparams["n_hits"],
                                                   majority_cut = self.hparams["majority_cut"],
                                                   primary = False)
        
        if log:
            
            self.log_dict(
                {
                    **tracking_performace_metrics,
                    "val_loss": loss,
                }
            )
        return bipartite_graph, loss

    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs[1]

    def test_step(self, batch, batch_idx):
        """
        Step to evaluate the model's performance
        """
        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs[1]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        """
        Use this to manually enforce warm-up. In the future, this may become built-into PyLightning
        """
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.global_step < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                if self.hparams["model"] == "mlp" or self.hparams["model"] == 3:
                    pg["lr"] = lr_scale * self.hparams["mlp_lr"]
                else:
                    pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()