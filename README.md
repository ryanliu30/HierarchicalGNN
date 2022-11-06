<div align="center">

# Hierarchical Graph Neural Network for Particle Track Reconstruction

<figure>
    <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/final_wide.png" width="250"/>
</figure>
    
### Exa.TrkX Collaboration


[ACAT 2022 Presentation](https://indico.cern.ch/event/1106990/contributions/4996236/)

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