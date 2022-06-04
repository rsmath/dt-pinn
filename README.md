# Accelerated Training of Physics Informed Neural Networks (PINNs) using Meshless Discretizations

This repository material contains the code for running and replicating our experiments for vanilla-PINNs and DT-PINNs.

## Introduction

Physics-informed neural networks (PINNs) are neural networks trained by using physical laws in the form of partial differential equations (PDEs) as soft constraints. We present a new technique for the accelerated training of PINNs that combines modern scientific computing techniques with machine learning: discretely-trained PINNs (DT-PINNs). The repeated computation of the partial derivative terms in the PINN loss functions via automatic differentiation during training is known to be computationally expensive, especially for higher-order derivatives. DT-PINNs are trained by replacing these exact spatial derivatives with high-order accurate numerical discretizations computed using meshless radial basis function-finite differences (RBF-FD) and applied via sparse-matrix vector multiplication. While in principle any high-order discretization may be used, the use of RBF-FD allows for DT-PINNs to be trained even on point cloud samples placed on irregular domain geometries. Additionally, though traditional PINNs (vanilla-PINNs) are typically stored and trained in 32-bit floating-point (fp32) on the GPU, we show that for DT-PINNs, using fp64 on the GPU leads to significantly faster training times than fp32 vanilla-PINNs with comparable accuracy. We demonstrate the efficiency and accuracy of DT-PINNs via a series of experiments. First, we explore the effect of network depth on both numerical and automatic differentiation of a neural network with random weights and show that RBF-FD approximations of third-order accuracy and above are more efficient while being sufficiently accurate. We then compare the DT-PINNs to vanilla-PINNs on both linear and nonlinear Poisson equations and show that DT-PINNs achieve similar losses with 2-4x faster training times on a consumer GPU. Finally, we also demonstrate that similar results can be obtained for the PINN solution to the heat equation (a space-time problem) by discretizing the spatial derivatives using RBF-FD and using automatic differentiation for the temporal derivative. Our results show that fp64 DT-PINNs offer a superior cost-accuracy profile to fp32 vanilla-PINNs, opening the door to a new paradigm of leveraging scientific computing techniques to support machine learning.

## Requirements

To run the Matlab code in the [``MatlabSolver/``](MatlabSolver/) folder, a Matlab account is required. For running the code for vanilla-PINNs and DT-PINNs, the following Python libraries are required:

1. numpy
2. matplotlib
3. json
4. scipy
5. cupy
6. torch

We provide a [``requirements.txt``](requirements.txt) file that can be used to install the libraries with pip:

```bash
>> git clone git@github.com:ramanshsharma2806/dt-pinn.git
>> cd dt-pinn
>> pip install -r requirements.txt
```

Our code is compatible with both CPU and GPU, however to replicate our GPU results from the paper (end of this README), we recommend using a GPU.


## Dataset
We provide the Matlab code in [``MatlabSolver/``](MatlabSolver/) to generate the dataset for linear and nonlinear Poisson equations, and the heat equation.

### Poisson equation
To generate the datset for the Poisson equation, please use the [``GenSCAIMats.m``](MatlabSolver/GenSCAIMats.m) file. Inside it, change the ``nonlinear`` variable value to 0 for linear and 1 for nonlinear. Running this file automatically makes the corresponding ``scai/`` (or ``nonlinear/`` in case of nonlinear Poisson) dataset folder.

### Heat equation
For the heat equation, use the [``GenerateVectorsTimeDependent.m``](MatlabSolver/GenerateVectorsTimeDependent.m) file. This file will add the three heat equation relevant vectors; ``u_heat.mat``, ``f_heat.mat``, and ``g_heat.mat`` in the dataset folder in all subfolders for size 828.

## PINN code
We provide the Python code in [``src/``](src/) for vanilla-PINN and DT-PINN for all experiments: linear and nonlinear Poisson, and the heat equation. While the names of the code modules are self-explanatory, we clarify some of them below:

1. [``dtpinn_cupy_fp32.py``](src/dtpinn_cupy_fp32.py): Corresponds fo fp32 DT-PINN for linear Poisson equation.
2. [``dtpinn_cupy_fp64.py``](src/dtpinn_cupy_fp64.py): Corresponds fo fp64 DT-PINN for linear Poisson equation.
3. [``dtpinn_cupy_fp64_nonlinear.py``](src/dtpinn_cupy_fp64_nonlinear.py): Corresponds fo fp64 DT-PINN for nonlinear Poisson equation.
4. [``heat_dtpinn_cupy.py``](src/heat_dtpinn_cupy.py): Corresponds fo fp64 DT-PINN for the heat equation.

The vanilla-PINN code files [``vanilla.py``](src/vanilla.py), [``vanilla_nonlinear.py``](src/vanilla_nonlinear.py), and [``vanilla_heat.py``](src/vanilla_heat.py) correspond to the linear and nonlinear Poisson equations, and heat equation respectively.

For all PINN code, one can change the learning rate, floating point precision, training set size, order of differentiation (for DT-PINN), network depth, and activation function directly in the code files. The code contains the hyperparameters we have used throughout our experiments. The files automatically save the results in a descriptive folder name that can later be used for making plots.

## Plots
Lastly, we also share a script [``plotting.py``](src/plotting.py) for making the plots from the paper for any of the generated results. Since usually the plots were made to compare DT-PINNs and vanilla-PINNs, the file contains two empty variables ``discrete_results_folder`` and ``vanilla_results_folder`` that are to be filled with the folders of the results. The different plots required can be chosen by uncommenting the function calls from the bottom of the file.

## Citation
This repository is part of the following paper. Please cite the following paper if you would like to cite this repository and find its contents useful:

```text
@article{SharmaShankar2022acceleratingpinn,
  title={Accelerated Training of Physics Informed Neural Networks (PINNs) using Meshless Discretizations},
  author = {Sharma, Ramansh and Shankar, Varun},
  url = {https://arxiv.org/abs/2205.09332},
  year={2022}
}
