<div align="center">

# Hierarchical Graph Neural Network for Particle Track Reconstruction

<figure>
    <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/final_wide.png" width="250"/>
</figure>
    
### Exa.TrkX Collaboration


[ACAT 2022 Presentation](https://indico.cern.ch/event/1106990/contributions/4996236/)
    
[arXiv paper](https://arxiv.org/abs/2303.01640)

[Author Contact](mailto:liuryan30@berkeley.edu)

</div>

Welcome to repository for Hierarchical Graph Neural Network for Particle Track Reconstruction. 

## Objectives

1. To present a Hierarchical GNN implementation to the HEP and ML community.
2. To present an example of using HGNN for particle tracking on the TrackML dataset.
3. To provide a quick comparison between HGNN and flat GNNs.

## Install

It's recommended to start a conda environment before installation:

```
conda create --name exatrkx-tracking python=3.8
conda activate exatrkx-tracking
pip install pip --upgrade
```

If you have a CUDA GPU available, load the toolkit or [install it](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) now. You should check that this is done by running `nvcc --version`. Then, install the following dependencies:

```
cudf                      22.04.00        
cugraph                   22.04.00        
cuml                      22.04.00       
cupy                      9.6.0            
frnn                      0.0.0                    
numba                     0.55.1           
numpy                     1.21.6 
pytorch-lightning         1.6.3
numpy                     1.21.6           
rapids                    22.04.00        
scikit-learn              1.0.2            
scipy                     1.8.0                                 
torch                     1.11.0+cu115             
torch-geometric           2.0.4                    
torch-scatter             2.0.9                    
torch-sparse              0.6.13                   
wandb                     0.12.16                  
yaml                      0.2.5                
```
Other combinations of version should also be working but not have tested yet.
Finally, clone the [repo](https://github.com/ryanliu30/HierarchicalGNN) and you are ready for quick-start.

## Usage

Once installation is done, download the quickstart TrackML 1GeV filter-processed [dataset](https://portal.nersc.gov/cfs/m3443/ryanliu/TrackML1GeV/events.tar.gz). Then, change the line `input_dir: /global/cfs/cdirs/m3443/data/trackml-codalab/gnn_processed/1GeV_pt_cut_quickstart_example` [here](https://github.com/ryanliu30/HierarchicalGNN/blob/e44edb8960d7f85a9d7562032fb26fb232efad79/Modules/BipartiteClassification/Configs/HGNN_GMM.yaml#L2) to the directory where you expand the file. After that, navigate to the [example notebook](https://github.com/ryanliu30/HierarchicalGNN/blob/main/Notebooks/example.ipynb). You should be able to see a variable called `ROOT_PATH`. Set it to be the directory you wish to keep model checkpoints and loggings. Then run the section **import** and **training a new model**. There will be an input box that you should enter model ID or model name. Use **4** to try bipartite classifier HGNN for our first try. After that, a `WandB` login token will be needed for logging purpose. Provide yours if you already have one or register a new account at [wandb](https://wandb.ai/). Or alternatively you can comment out `logger = WandbLogger(project="TrackML_1GeV")` and change it to `logger = None` to disable logging. The model requires quite a lot GPU memory to run, so if cuda runs out of memory, navigate to [configs](https://github.com/ryanliu30/HierarchicalGNN/tree/main/Modules/BipartiteClassification/Configs) and change `latent` till the model fits to your GPU. Some advanced training techniques (e.g. multiple GPU training) can be found [here](https://github.com/ryanliu30/Tracking-ML-Exa.TrkX/blob/master/Pipelines/Common_Tracking_Example/notebooks/TrackML_ACAT/train_gnn.py) but note that we have observed some issues about multi-gpu training and are still working on to stablize it.
