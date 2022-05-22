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
from cupy.sparse import csr_matrix
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
L_t, B_t = None, None


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
        
class Cupy_mul_B(torch.autograd.Function):
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
        return from_dlpack(B_t.dot(cupy.from_dlpack(to_dlpack(grad_output))).toDlpack()), None

class Trainer:
    def __init__(self, config=None, **kwargs):
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.__dict__.update(kwargs)
        self.logged_results = defaultdict(list)

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
        global L_t, B_t
        epochs = self.epochs
        self.u_true_training = self.u_true[:self.x_tilde.shape[0], :].detach().clone()
        self.u_true_training.requires_grad = True

        # multiplying L and B with random vectors to "generate a kernel" and move them to the GPU
        rand_vec = cupy.from_dlpack(to_dlpack(torch.rand(self.L.shape[1], 2).to(torch.float64).to(device_string)))
        self.L.dot(rand_vec)
        self.B.dot(rand_vec)
        
        L_t = csr_matrix(self.L.transpose(), dtype=np.float64)
        B_t = csr_matrix(self.B.transpose(), dtype=np.float64)
        
        # initializing the matvec kernel for the sparse matrices' transposes
        rand_L_vec = cupy.from_dlpack(to_dlpack(torch.rand(self.L.shape[0], 2).to(torch.float64).to(device_string)))
        rand_B_vec = cupy.from_dlpack(to_dlpack(torch.rand(self.B.shape[0], 2).to(torch.float64).to(device_string)))
        L_t.dot(rand_L_vec)
        B_t.dot(rand_B_vec)
        
        L_mul = Cupy_mul_L.apply
        B_mul = Cupy_mul_B.apply

        time_range = torch.linspace(0, 1, 25)

        T_tilde = []
        for i in range(25):
            T_tilde.append(torch.ones(self.x_tilde.shape[0], 1) * time_range[i])

        T_tilde = torch.cat([*T_tilde], dim=0).reshape(25, self.x_tilde.shape[0], 1)
        T_tilde.requires_grad = True
        T_tilde = T_tilde.to(device_string)
        
        # for boundary points
        T_b = []
        for i in range(25):
            T_b.append(torch.ones(self.X_b.shape[0], 1) * time_range[i])

        T_b = torch.cat([*T_b], dim=0).reshape(25, self.X_b.shape[0], 1)
        T_b.requires_grad = True
        T_b = T_b.to(device_string)
        
        # including ghost points
        T_full = []
        for i in range(25):
            T_full.append(torch.ones(self.X_full.shape[0], 1) * time_range[i])

        T_full = torch.cat([*T_full], dim=0).reshape(25, self.X_full.shape[0], 1)
        T_full.requires_grad = True
        T_full = T_full.to(device_string)
        
        if device_string == "cuda":
            torch.cuda.synchronize() # first call to get all cuda tensors on GPU
        start = time.perf_counter()

        for training_iter in range(1, epochs + 1):
            try:
                def closure():
                    self.optimizer.zero_grad()

                    f_pred_list = []
                    boundary_loss_list = []

                    for i in range(25):
                        cur_t = T_full[i, :, :]
                        nn_input = torch.cat([self.X_full, cur_t], dim=1)
                        cur_t_tilde = T_tilde[i, :, :]
                        nn_input_tilde = torch.cat([self.X_tilde, cur_t_tilde], dim=1)

                        # pred term on all points
                        u_pred_full = self.w.forward(nn_input)
                        
                        # pred term on tilde points
                        u_pred_tilde = self.w.forward(nn_input_tilde)

                        u_t = grad(u_pred_tilde, cur_t_tilde, grad_outputs=torch.ones_like(u_pred_tilde), create_graph=True, retain_graph=True)[0]

                        lap = L_mul(u_pred_full, self.L)

                        # Poisson residual
                        f_pred = u_t - lap
                        f_pred_list.append(f_pred.flatten())

                        # boundary residual
                        boundary_loss_term = B_mul(u_pred_full, self.B)

                        # first partial derivatives on the boundary
                        boundary_loss_list.append(boundary_loss_term.flatten())

                        if i == 0:
                            supervised_term = torch.mean(torch.square(u_pred_tilde - self.u_true_training))

                    l2 = torch.mean(torch.square(torch.cat([*f_pred_list], dim=0).reshape(self.f.shape[0], 1) - self.f))
                    l3 = torch.mean(torch.square(torch.cat([*boundary_loss_list], dim=0).reshape(self.g.shape[0], 1) - self.g))

                    train_loss = l2 + l3 + supervised_term

                    train_loss.backward(retain_graph=True)

                    return train_loss.item()

                loss_value = self.optimizer.step(closure)
                if device_string == "cuda":
                    torch.cuda.synchronize() # second call right before time clock to finish all operations
                epoch_time = time.perf_counter() - start

                '''
                logging errors and printing them
                '''
                training_pred = []
                pde_residual_training = []
                boundary_residual_training = []
                for i in range(25):
                    cur_t = T_full[i, :, :]
                    nn_input = torch.cat([self.X_full, cur_t], dim=1)
                    cur_t_tilde = T_tilde[i, :, :]
                    nn_input_tilde = torch.cat([self.X_tilde, cur_t_tilde], dim=1)
                        
                    cur_training_pred = self.w.forward(nn_input)
                    cur_training_tilde = self.w.forward(nn_input_tilde)
                    
                    u_t = grad(cur_training_tilde, cur_t_tilde, grad_outputs=torch.ones_like(cur_training_tilde), create_graph=True, retain_graph=True)[0]

                    lap = L_mul(cur_training_pred, self.L)
                    assert lap.dtype == torch.float64

                    # Poisson residual
                    f_pred = u_t - lap
                    pde_residual_training.append(f_pred.flatten())

                    # boundary residual
                    boundary_loss_term = B_mul(cur_training_pred, self.B)
                    assert boundary_loss_term.dtype == torch.float64

                    boundary_residual_training.append(boundary_loss_term.flatten())
                    training_pred.append(cur_training_tilde)

                training_pred = torch.cat([*training_pred], dim=0).reshape(self.u_true.shape[0], 1)
                pde_residual_training = torch.cat([*pde_residual_training], dim=0).reshape(self.u_true.shape[0], 1) - self.f
                boundary_residual_training = torch.cat([*boundary_residual_training], dim=0).reshape(self.g.shape[0], 1) - self.g


                # storing losses in variables
                training_pde_residual = self.compute_linf(pde_residual_training)
                training_boundary_residual = self.compute_linf(boundary_residual_training)
                training_mse = self.compute_mse(training_pred, self.u_true)
                training_l2 = self.compute_l2(training_pred, self.u_true)

                # logging losses and other important things in lists
                self.logged_results['training_losses'].append(loss_value)
                self.logged_results['training_mse_losses'].append(training_mse)
                self.logged_results['training_l2_losses'].append(training_l2)
                self.logged_results['training_pde_residual'].append(training_pde_residual)
                self.logged_results['training_boundary_residual'].append(training_boundary_residual)
                self.logged_results['epochs_list'].append(training_iter)
                self.logged_results['epoch_time'].append(epoch_time)
                
                # Anneal learning rate if the loss is more than moving average of previous 10 elements
                annealing_count = 0
                annealing_counter = 0
                if annealing_count > 10 and loss_value > sum(self.logged_results['training_losses'][-11:-1])/10.:
                    annealing_count = 0
                    annealing_counter += 1
                    print('\n' + '-'*40)
                    print('Annealing learning rate.')
                    self.optimizer.param_groups[0]['lr'] -= 0.025
                    print(f"New learning rate is: {self.optimizer.param_groups[0]['lr']}")
                    print('-'*40 + '\n')
                annealing_count += 1

                if training_iter > 30 and (isnan(loss_value) or loss_value > 500):
                    # loss has exploded, this will trigger this run to restart
                    print(f"Loss exploded to: {loss_value}")
                    return False

                if training_iter % self.print_interval == 0 and self.verbosity:
                    print('='*70)
                    print(f'Iter {training_iter},\nTraining loss = {loss_value}')

                    # laplacian(pinn) - f
                    print('\nu_t - Lu - f')
                    print('===TRAINING' + '='*36)
                    print(training_pde_residual)

                    # alpha * d/dn pinn + beta * pinn - g
                    print('\nalpha * d/dn pinn + beta * pinn - g')
                    print('===TRAINING' + '='*36)
                    print(training_boundary_residual)

                    # mse errors
                    print('\nMSE errors w.r.t. u_true')
                    # training
                    print('===TRAINING' + '='*36)
                    print(f'PINN MSE: {training_mse}')

                    # l2 errors
                    print(f'\nL2 errors w.r.t. u_true')
                    # training
                    print('===TRAINING' + '='*36)
                    print(f'PINN L2: {training_l2}')
            except KeyboardInterrupt:
                print('Keyboard Interrupt. Ending training.')
                return dict(self.logged_results)

        print(f"Learning rate annealed {annealing_counter} times")
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
    device_string = "cuda"
    network_precisions = ["float64"]
    orders = [2, 3, 4, 5]
    training_size = [828]
    supervised_options = [False]
    activation_function = 'tanh'
    lr = 0.4

    for network_precision in network_precisions:
        results_folder = f"gpu_heat_{network_precision}_cupy_csr_results_noprint"
        print(f"\nUsing network precision:{network_precision}")
        for cur_supervised_option in supervised_options:
            print(f'\nGoing over supervised={cur_supervised_option}')
            for order in orders:
                for size in training_size:
                    supervised_file_bool = 'supervised' if cur_supervised_option else 'unsupervised'
                    file_name = f"{order}_{size}"
                    save_folder = f'../{results_folder}/discrete_pinn/{order}/{size}/{supervised_file_bool}/'
                    if not os.path.isdir(save_folder):
                        os.makedirs(save_folder)
                    save_file_name = save_folder + 'results.json'

                    print('\n\n')
                    print('+'*70)
                    print(f'Running heat {network_precision} DT-PINN on {file_name} with {activation_function}')
                    print('+'*70)
                    print('\n\n')

                    # read mat files
                    X_i = torch.tensor(loadmat(f"../scai/files_{file_name}/Xi.mat")["Xi"], dtype=PRECISION, requires_grad=True).to(device_string)
                    X_b = torch.tensor(loadmat(f"../scai/files_{file_name}/Xb.mat")["Xb"], dtype=PRECISION, requires_grad=True).to(device_string)
                    X_g = torch.tensor(loadmat(f"../scai/files_{file_name}/Xg.mat")["X_g"], dtype=PRECISION, requires_grad=True).to(device_string)
                    n = torch.tensor(loadmat(f"../scai/files_{file_name}/n.mat")["n"], dtype=PRECISION, requires_grad=True).to(device_string)
                    u_true = torch.tensor(loadmat(f"../scai/files_{file_name}/u_heat.mat")["u"], dtype=PRECISION).to(device_string)
                    f = torch.tensor(loadmat(f"../scai/files_{file_name}/f_heat.mat")["f"], dtype=PRECISION, requires_grad=True).to(device_string)
                    g = torch.tensor(loadmat(f"../scai/files_{file_name}/g_heat.mat")["g"], dtype=PRECISION, requires_grad=True).to(device_string)
                    L = load_mat_cupy(loadmat(f"../scai/files_{file_name}/L1.mat")["L1"])
                    B = load_mat_cupy(loadmat(f"../scai/files_{file_name}/B1.mat")["B1"])
                    alpha = torch.tensor(loadmat(f"../scai/files_{file_name}/alpha.mat")["Neucoeff"], dtype=PRECISION, requires_grad=True).to(device_string)
                    beta = torch.tensor(loadmat(f"../scai/files_{file_name}/beta.mat")["Dircoeff"], dtype=PRECISION).to(device_string)
                    b_starts = X_i.shape[0]
                    b_end = b_starts + X_b.shape[0]


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
                        'spatial_dim': 3,
                        'precision': network_precision,
                        'activation': activation_function,
                        'network_device': device_string,
                        'order': 2, # activation order
                        'layers': 4,
                        'nodes': 50,
                        'epochs': 5000,
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
                        'L': L,
                        'B': B,
                        'u_true': u_true,
                        'f': f,
                        'g': g,
                        'alpha': alpha,
                        'beta': beta,
                        'b_end': b_end,
                        'b_starts': b_starts,
                        'supervised': cur_supervised_option,
                        'print_interval': 1,
                        'verbosity': True,
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

