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

If you have a CUDA GPU available, load the toolkit or [install it](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) now. You should check that this is done by running `nvcc --version`. Then, intall the following dependencies:

```
cudf                      22.04.00        cuda_11_py38_g8bf0520170_0    rapidsai
cugraph                   22.04.00        cuda11_py38_g58be5b53_0    rapidsai
cuml                      22.04.00        cuda11_py38_g95abbc746_0    rapidsai
cupy                      9.6.0            py38h177b0fd_0    conda-forge
frnn                      0.0.0                    pypi_0    pypi
numba                     0.55.1           py38h4bf6c61_0    conda-forge
numpy                     1.21.6           py38h1d589f8_0    conda-forge
rapids                    22.04.00        cuda11_py38_ge08d166_149    rapidsai
scikit-learn              1.0.2            py38h1561384_0    conda-forge
scipy                     1.8.0            py38h56a6a73_1    conda-forge
sklearn                   0.0                      pypi_0    pypi
torch                     1.11.0+cu115             pypi_0    pypi
torch-geometric           2.0.4                    pypi_0    pypi
torch-scatter             2.0.9                    pypi_0    pypi
torch-sparse              0.6.13                   pypi_0    pypi
wandb                     0.12.16                  pypi_0    pypi
yaml                      0.2.5                h7f98852_2    conda-forge
```
