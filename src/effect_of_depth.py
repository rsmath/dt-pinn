import json
from collections import defaultdict
import os
import math
from datetime import datetime
from math import log, floor, isnan
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim.lr_scheduler as schedulers
from torch import optim
from torch import nn
import torch
from torch.nn import MSELoss
from network import W
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import griddata
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

torch.manual_seed(0)

# CUDA support
if torch.cuda.is_available():
    pytorch_device = torch.device('cuda')
    torch.cuda.init()
    device_string = "cuda"
    torch.cuda.manual_seed_all(0)
else:
    pytorch_device = torch.device('cpu')
    device_string = "cpu"
print(f"Device being used: {device_string}")

# global transposes of the sparse matrices
L_t= None


class Cupy_mul_L(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_pred_, sparse):
        """
        u_pred is the network's prediction
        """
        return from_dlpack(sparse.dot(cupy.from_dlpack(to_dlpack(u_pred_))).toDlpack())
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output is with respect to u_pred
        """
        return from_dlpack(L_t.dot(cupy.from_dlpack(to_dlpack(grad_output))).toDlpack()), None
        
class Trainer:
    def __init__(self, config=None, **kwargs):
        self.lr = config['lr']
        self.network_precision_string = config['precision']
        self.network_precision_dtype = torch.float32 if self.network_precision_string == "float32" else torch.float64
        self.epochs = config['epochs']
        self.__dict__.update(kwargs)
        self.logged_results = defaultdict(list)
        self.f_pred_ = None
        self.boundary_loss_term_ = None

        self.w = W(config)
        if self.verbosity:
            print(self.w, '\n')

        # interior points
        self.x_interior = torch.vstack([self.x_i.clone()])
        self.y_interior = torch.vstack([self.y_i.clone()])
        self.X_interior = torch.hstack([self.x_interior, self.y_interior])

        # interior and boundary points
        self.x_tilde = torch.vstack([self.x_i.clone(), self.x_b.clone()])
        self.y_tilde = torch.vstack([self.y_i.clone(), self.y_b.clone()])
        self.X_tilde = torch.hstack([self.x_tilde, self.y_tilde])

        # interior, boundary, and ghost points
        self.x_full = torch.vstack([self.x_i.clone(), self.x_b.clone(), self.x_g.clone()])
        self.y_full = torch.vstack([self.y_i.clone(), self.y_b.clone(), self.y_g.clone()])
        self.X_full = torch.hstack([self.x_full, self.y_full])

        # boundary points
        self.X_b = torch.hstack([self.x_b.clone(), self.y_b.clone()])

        if config['optimizer'] == 'adam':
            self.optimizer_choice = optim.Adam
        elif config['optimizer'] == 'lbfgs':
            self.optimizer_choice = optim.LBFGS
        elif config['optimizer'] == 'sgd':
            self.optimizer_choice = optim.SGD

        self.optimizer = self.optimizer_choice(self.w.parameters(), lr=self.lr)

    @staticmethod
    def compute_mse(a, b):
        mse = MSELoss()(torch.flatten(a), torch.flatten(b))
        return mse.item()

    @staticmethod
    def compute_l2(a, b):
        diff = torch.subtract(torch.flatten(a).detach().cpu(), torch.flatten(b).detach().cpu())
        relative_l2_error = torch.linalg.norm(diff) / torch.linalg.norm(torch.flatten(b))
        return relative_l2_error.item()

    @staticmethod
    def compute_linf(a):
        return torch.linalg.norm(a.to(PRECISION), ord=float('inf')).item()

    def train(self):
        # multiplying L and B with random vectors to "generate a kernel" and move them to the GPU
        rand_vec = cupy.from_dlpack(to_dlpack(torch.rand(self.L.shape[1], 2).to(torch.float64).to(device_string)))
        self.L.dot(rand_vec)
        L_mul = Cupy_mul_L.apply

        # discrete
        u_pred_full = self.w.forward(self.X_full)
        
        assert self.X_full.dtype == torch.float64
        assert u_pred_full.dtype == torch.float64

        if device_string == "cuda":
            torch.cuda.synchronize() # first call to get all cuda tensors on GPU
        start = time.perf_counter()
        
        discrete_value = L_mul(u_pred_full, self.L)

        if device_string == "cuda":
            torch.cuda.synchronize() # second call right before time clock to finish all operations
        discrete_time = time.perf_counter() - start
        
        # fp64 autograd
        u_pred_tilde = self.w.forward(self.X_tilde)
        
        if device_string == "cuda":
            torch.cuda.synchronize() # first call to get all cuda tensors on GPU
        start = time.perf_counter()
        
        u_x = grad(u_pred_tilde, self.x_tilde, grad_outputs=torch.ones_like(
                   u_pred_tilde), create_graph=True, retain_graph=True)[0]
        u_xx = grad(u_x, self.x_tilde, grad_outputs=torch.ones_like(
                    u_pred_tilde), create_graph=True, retain_graph=True)[0]

        u_y = grad(u_pred_tilde, self.y_tilde, grad_outputs=torch.ones_like(
                   u_pred_tilde), create_graph=True, retain_graph=True)[0]
        u_yy = grad(u_y, self.y_tilde, grad_outputs=torch.ones_like(
                    u_pred_tilde), create_graph=True, retain_graph=True)[0]

        # Poisson residual
        fp64_autograd_value = (u_xx + u_yy)
        
        if device_string == "cuda":
            torch.cuda.synchronize() # second call right before time clock to finish all operations
        fp64_autograd_time = time.perf_counter() - start

        # fp32 autograd
        self.x_tilde = self.x_tilde.to(torch.float32)
        self.y_tilde = self.y_tilde.to(torch.float32)
        self.X_tilde = torch.hstack([self.x_tilde, self.y_tilde])
        
        config['precision'] = 'float32'
        self.w = W(config)
        
        u_pred_tilde = self.w.forward(self.X_tilde)
        
        assert self.x_tilde.dtype == torch.float32
        assert self.y_tilde.dtype == torch.float32
        assert u_pred_tilde.dtype == torch.float32

        if device_string == "cuda":
            torch.cuda.synchronize() # first call to get all cuda tensors on GPU
        start = time.perf_counter()

        u_x = grad(u_pred_tilde, self.x_tilde, grad_outputs=torch.ones_like(
                   u_pred_tilde), create_graph=True, retain_graph=True)[0]
        u_xx = grad(u_x, self.x_tilde, grad_outputs=torch.ones_like(
                    u_pred_tilde), create_graph=True, retain_graph=True)[0]

        u_y = grad(u_pred_tilde, self.y_tilde, grad_outputs=torch.ones_like(
                   u_pred_tilde), create_graph=True, retain_graph=True)[0]
        u_yy = grad(u_y, self.y_tilde, grad_outputs=torch.ones_like(
                    u_pred_tilde), create_graph=True, retain_graph=True)[0]

        # Poisson residual
        fp32_autograd_value = (u_xx + u_yy)

        if device_string == "cuda":
            torch.cuda.synchronize() # second call right before time clock to finish all operations
        fp32_autograd_time = time.perf_counter() - start

        discrete_fp64_autograd_l2 = self.compute_l2(discrete_value, fp64_autograd_value)
        fp32_autograd_fp64_autograd_l2 = self.compute_l2(fp32_autograd_value, fp64_autograd_value)

        # storing losses in variables
        self.logged_results['discrete_time'].append(discrete_time)
        self.logged_results['fp32_autograd_time'].append(fp32_autograd_time)
        self.logged_results['fp64_autograd_time'].append(fp64_autograd_time)
        self.logged_results['discrete_fp64_autograd_l2'].append(discrete_fp64_autograd_l2)
        self.logged_results['fp32_autograd_fp64_autograd_l2'].append(fp32_autograd_fp64_autograd_l2)

        return dict(self.logged_results)

def load_mat_cupy(mat):
    csr = csr_matrix(mat, dtype=np.float64)
    return csr

def save_results(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # cupy setup
    device = cupy.cuda.Device(0)
    cupy.cuda.Device(0).use()
    
    PRECISION = torch.float64
    network_precisions = ["float64"]
    orders = [2, 3, 4, 5]
    layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # just one training set size
    size = 19638
    supervised_options = [False]
    activation_function = 'tanh'
    lr = 0.04

    for network_precision in network_precisions:
        results_folder = f"gpu_effect_depth_discrete_{network_precision}_vs_fp32_fp64_autograd"
        print(f"\nUsing network precision:{network_precision}")
        for cur_supervised_option in supervised_options:
            print(f'\nGoing over supervised={cur_supervised_option}')
            for layer in layers:
                print(f"Going over layer={layer}")
                for order in orders:
                    supervised_file_bool = 'supervised' if cur_supervised_option else 'unsupervised'
                    file_name = f"{order}_{size}"
                    test_name = f"{order}_21748_test"
                    save_folder = f'../{results_folder}/discrete_pinn/{order}/{size}/{layer}/{supervised_file_bool}/'
                    if not os.path.isdir(save_folder):
                        os.makedirs(save_folder)
                    save_file_name = save_folder + 'results.json'

                    print('\n\n')
                    print('+'*70)
                    print(f'Finding effect of neural network depth {file_name} with {activation_function}')
                    print('+'*70)
                    print('\n\n')

                    # read mat files
                    X_i = torch.tensor(loadmat(f"../scai/files_{file_name}/Xi.mat")["Xi"], dtype=PRECISION, requires_grad=True).to(device_string)
                    X_b = torch.tensor(loadmat(f"../scai/files_{file_name}/Xb.mat")["Xb"], dtype=PRECISION, requires_grad=True).to(device_string)
                    X_g = torch.tensor(loadmat(f"../scai/files_{file_name}/Xg.mat")["X_g"], dtype=PRECISION, requires_grad=True).to(device_string)
                    n = torch.tensor(loadmat(f"../scai/files_{file_name}/n.mat")["n"], dtype=PRECISION, requires_grad=True).to(device_string)
                    u_true = torch.tensor(loadmat(f"../scai/files_{file_name}/u.mat")["u"], dtype=PRECISION).to(device_string)
                    f = torch.tensor(loadmat(f"../scai/files_{file_name}/f.mat")["f"], dtype=PRECISION, requires_grad=True).to(device_string)
                    g = torch.tensor(loadmat(f"../scai/files_{file_name}/g.mat")["g"], dtype=PRECISION, requires_grad=True).to(device_string)
                    alpha = torch.tensor(loadmat(f"../scai/files_{file_name}/alpha.mat")["Neucoeff"], dtype=PRECISION, requires_grad=True).to(device_string)
                    beta = torch.tensor(loadmat(f"../scai/files_{file_name}/beta.mat")["Dircoeff"], dtype=PRECISION, requires_grad=True).to(device_string)
                    L = load_mat_cupy(loadmat(f"../scai/files_{file_name}/L1.mat")["L1"])
                    B = load_mat_cupy(loadmat(f"../scai/files_{file_name}/B1.mat")["B1"])
                    time.sleep(1)
                    b_starts = X_i.shape[0]
                    b_end = b_starts + X_b.shape[0]

                    # test files
                    X_i_test = torch.tensor(loadmat(f"../scai/files_{test_name}/Xi.mat")["Xi"], dtype=PRECISION, requires_grad=True).to(device_string)
                    X_b_test = torch.tensor(loadmat(f"../scai/files_{test_name}/Xb.mat")["Xb"], dtype=PRECISION, requires_grad=True).to(device_string)
                    test_u_true = torch.tensor(loadmat(f"../scai/files_{test_name}/u.mat")["u"], dtype=PRECISION).to(device_string)
                    f_test = torch.tensor(loadmat(f"../scai/files_{test_name}/f.mat")["f"], dtype=PRECISION, requires_grad=True).to(device_string)
                    g_test = torch.tensor(loadmat(f"../scai/files_{test_name}/g.mat")["g"], dtype=PRECISION, requires_grad=True).to(device_string)
                    alpha_test = torch.tensor(loadmat(f"../scai/files_{test_name}/alpha.mat")["Neucoeff"], dtype=PRECISION, requires_grad=True).to(device_string)
                    beta_test = torch.tensor(loadmat(f"../scai/files_{test_name}/beta.mat")["Dircoeff"], dtype=PRECISION, requires_grad=True).to(device_string)
                    n_test = torch.tensor(loadmat(f"../scai/files_{test_name}/n.mat")["n"], dtype=PRECISION, requires_grad=True).to(device_string)

                    test_x_i = X_i_test[:, 0].unsqueeze(dim=1)
                    test_y_i = X_i_test[:, 1].unsqueeze(dim=1)
                    test_x_b = X_b_test[:, 0].unsqueeze(dim=1)
                    test_y_b = X_b_test[:, 1].unsqueeze(dim=1)

                    test_x_tilde = torch.vstack([test_x_i, test_x_b])
                    test_y_tilde = torch.vstack([test_y_i, test_y_b])
                    test_X_tilde = torch.hstack([test_x_tilde, test_y_tilde])

                    test_X_b = torch.hstack([test_x_b, test_y_b])

                    # need to separate the spatial dimensions in X matrices for proper partial derivatives with autograd
                    x_i = X_i[:, 0].unsqueeze(dim=1)
                    y_i = X_i[:, 1].unsqueeze(dim=1)
                    x_b = X_b[:, 0].unsqueeze(dim=1)
                    y_b = X_b[:, 1].unsqueeze(dim=1)
                    x_g = X_g[:, 0].unsqueeze(dim=1)
                    y_g = X_g[:, 1].unsqueeze(dim=1)

                    # only compute losses on interior and boundary points
                    ib_idx = X_i.shape[0] + X_b.shape[0]

                    # define list for Trainer input
                    config = {
                        'spatial_dim': 2,
                        'precision': network_precision,
                        'activation': activation_function,
                        'order': 2, # activation order
                        'network_device': device_string,
                        'layers': layer,
                        'nodes': 50,
                        'epochs': 1,
                        'optimizer': 'lbfgs',
                        'lr': lr,
                    }
                    print(f"Learning rate: {config['lr']}")
                    vars = {
                        'n': n,
                        'x_i': x_i,
                        'x_b': x_b,
                        'x_g': x_g,
                        'y_i': y_i,
                        'y_b': y_b,
                        'y_g': y_g,
                        'ib_idx': ib_idx,
                        'u_true': u_true,
                        'L': L,
                        'B': B,
                        'test_X_tilde': test_X_tilde,
                        'test_u_true': test_u_true,
                        'test_x_i': test_x_i,
                        'test_y_i': test_y_i,
                        'test_x_b': test_x_b,
                        'test_y_b': test_y_b,
                        'test_x_tilde': test_x_tilde,
                        'test_y_tilde': test_y_tilde,
                        'test_X_b': test_X_b,
                        'f_test': f_test,
                        'beta_test': beta_test,
                        'alpha_test': alpha_test,
                        'g_test': g_test,
                        'n_test': n_test,
                        'f': f,
                        'g': g,
                        'alpha': alpha,
                        'beta': beta,
                        'b_end': b_end,
                        'b_starts': b_starts,
                        'supervised': cur_supervised_option,
                        'print_interval': 10,
                        'verbosity': False,
                    }

                    flag = True
                    while flag:
                        trainer = Trainer(config=config, **vars)
                        logged_results = trainer.train()
                        if type(logged_results) == bool:
                            config['lr'] /= 2.0
                            print(f"Restarting with learning rate = {config['lr']}")
                            continue
                        else:
                            flag = False

                    logged_results = logged_results | config
                    save_results(logged_results, save_file_name)
                    config['lr'] = lr

