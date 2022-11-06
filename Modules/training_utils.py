import sys
import yaml
sys.path.append("../Modules/")
from EdgeClassifier.Models.IN import EC_InteractionGNN
from GNNEmbedding.Models.IN import Embedding_InteractionGNN
from GNNEmbedding.Models.HGNN_GMM import Embedding_HierarchicalGNN_GMM
from BipartiteClassification.Models.HGNN_GMM import BC_HierarchicalGNN_GMM

path = "../Modules/"

def process_hparams(hparams):    
    if hparams["hidden"] == "ratio":
        hparams["hidden"] = hparams["hidden_ratio"]*hparams["latent"]
    
    if "cluster_granularity" not in hparams:
        hparams["cluster_granularity"] = 0
    
    return hparams

def model_selector(model_name, sweep_configs = {}):
    if model_name == "EC-IN" or model_name == "1":
        with open(path + "EdgeClassifier/Configs/IN.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader)
        model = EC_InteractionGNN(process_hparams({**hparams, **sweep_configs}))
    elif model_name == "Embedding-IN" or model_name == "2":
        with open(path + "GNNEmbedding/Configs/IN.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader) 
        model = Embedding_InteractionGNN(process_hparams({**hparams, **sweep_configs}))
    elif model_name == "Embedding-HGNN-GMM" or model_name == "3":
        with open(path + "GNNEmbedding/Configs/HGNN_GMM.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader) 
        model = Embedding_HierarchicalGNN_GMM(process_hparams({**hparams, **sweep_configs}))
    elif model_name == "BC-HGNN-GMM" or model_name == "4":
        with open(path + "BipartiteClassification/Configs/HGNN_GMM.yaml") as f:
            hparams = yaml.load(f, Loader=yaml.FullLoader) 
        model = BC_HierarchicalGNN_GMM(process_hparams({**hparams, **sweep_configs}))      
    else:
        raise ValueError("Can't Find Model Name {}!".format(model_name))
        
    return model
