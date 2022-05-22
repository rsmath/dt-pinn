import json
import math
import os
from datetime import datetime
from math import log, floor, isnan
from collections import defaultdict
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
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

torch.manual_seed(0)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
    pytorch_device = torch.device('cuda')
    device_string = "cuda"
    torch.cuda.manual_seed_all(0)
else:
    device = torch.device('cpu')
    device_string = "cpu"
print(f"Device being used: {device}")


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
        epochs = self.epochs
        self.u_tilde = self.u[:self.ib_idx] # interior + boundary
        self.f_interior = self.f[:self.b_starts] # only interior

        if device_string == "cuda":
            torch.cuda.synchronize() # first call to get all cuda tensors on GPU
        start = time.perf_counter()

        for i in range(1, epochs + 1):
            try:
                def closure():
                    self.optimizer.zero_grad()

                    # residual term on interior points only
                    u_pred_interior = self.w.forward(self.X_interior)
                    u_x = grad(u_pred_interior, self.x_interior, grad_outputs=torch.ones_like(
                               u_pred_interior), create_graph=True, retain_graph=True)[0]
                    u_xx = grad(u_x, self.x_interior, grad_outputs=torch.ones_like(
                                u_pred_interior), create_graph=True, retain_graph=True)[0]

                    u_y = grad(u_pred_interior, self.y_interior, grad_outputs=torch.ones_like(
                               u_pred_interior), create_graph=True, retain_graph=True)[0]
                    u_yy = grad(u_y, self.y_interior, grad_outputs=torch.ones_like(
                                u_pred_interior), create_graph=True, retain_graph=True)[0]

                    # Poisson residual
                    f_pred = (u_xx + u_yy) - self.f_interior

                    boundary_pred = self.w.forward(self.X_b)

                    # first partial derivatives on the boundary
                    l2_w_x = grad(boundary_pred, self.x_b, grad_outputs=torch.ones_like(boundary_pred),
                                  create_graph=True, retain_graph=True)[0]
                    l2_w_y = grad(boundary_pred, self.y_b, grad_outputs=torch.ones_like(boundary_pred),
                                  create_graph=True, retain_graph=True)[0]

                    # combining the partial derivatives
                    w_xy = torch.hstack([l2_w_x, l2_w_y])

                    # element wise dot product between the normal vectors and first partial derivatives
                    gradient_n = torch.multiply(self.n, w_xy).sum(dim=1).unsqueeze(dim=1)

                    # loss on the boundary
                    boundary_loss_term = torch.multiply(self.alpha, gradient_n) + \
                                         torch.multiply(self.beta, boundary_pred) - self.g
                    l2 = torch.mean(torch.square(torch.flatten(f_pred)))
                    l3 = torch.mean(torch.square(torch.flatten(boundary_loss_term)))

                    train_loss = l2 + l3
                    train_loss.backward(retain_graph=True)
                    return train_loss.item()

                loss_value = self.optimizer.step(closure)
                if device_string == "cuda":
                    torch.cuda.synchronize() # second call right before time clock to finish all operations
                epoch_time = time.perf_counter() - start

                '''
                logging errors and printing them
                '''
                training_pred = self.w.forward(self.X_tilde)
                test_pred = self.w.forward(self.test_X_tilde)

                # TRAINING
                pred_tilde = self.w.forward(self.X_tilde)
                u_x = grad(pred_tilde, self.x_tilde, grad_outputs=torch.ones_like(
                           pred_tilde), create_graph=True, retain_graph=True)[0]
                u_xx = grad(u_x, self.x_tilde, grad_outputs=torch.ones_like(
                            pred_tilde), create_graph=True, retain_graph=True)[0]

                u_y = grad(pred_tilde, self.y_tilde, grad_outputs=torch.ones_like(
                           pred_tilde), create_graph=True, retain_graph=True)[0]
                u_yy = grad(u_y, self.y_tilde, grad_outputs=torch.ones_like(
                            pred_tilde), create_graph=True, retain_graph=True)[0]

                # Poisson residual
                pde_residual_training = (u_xx + u_yy) - self.f

                boundary_pred = self.w.forward(self.X_b)

                # first partial derivatives on the boundary
                l2_w_x = grad(boundary_pred, self.x_b, grad_outputs=torch.ones_like(boundary_pred),
                              create_graph=True, retain_graph=True)[0]
                l2_w_y = grad(boundary_pred, self.y_b, grad_outputs=torch.ones_like(boundary_pred),
                              create_graph=True, retain_graph=True)[0]

                # combining the partial derivatives
                w_xy = torch.hstack([l2_w_x, l2_w_y])

                # element wise dot product between the normal vectors and first partial derivatives
                gradient_n = torch.multiply(self.n, w_xy).sum(dim=1).unsqueeze(dim=1)

                # loss on the boundary
                boundary_residual_training = torch.multiply(self.alpha, gradient_n) + \
                                    torch.multiply(self.beta, boundary_pred) - self.g
                # TEST
                pred_tilde_test = self.w.forward(self.test_X_tilde)
                u_x = grad(pred_tilde_test, self.test_x_tilde, grad_outputs=torch.ones_like(
                           pred_tilde_test), create_graph=True, retain_graph=True)[0]
                u_xx = grad(u_x, self.test_x_tilde, grad_outputs=torch.ones_like(
                            pred_tilde_test), create_graph=True, retain_graph=True)[0]

                u_y = grad(pred_tilde_test, self.test_y_tilde, grad_outputs=torch.ones_like(
                           pred_tilde_test), create_graph=True, retain_graph=True)[0]
                u_yy = grad(u_y, self.test_y_tilde, grad_outputs=torch.ones_like(
                            pred_tilde_test), create_graph=True, retain_graph=True)[0]

                # Poisson residual
                pde_residual_test = (u_xx + u_yy) - self.f_test

                boundary_pred = self.w.forward(self.test_X_b)

                # first partial derivatives on the boundary
                l2_w_x = grad(boundary_pred, self.test_x_b, grad_outputs=torch.ones_like(boundary_pred),
                              create_graph=True, retain_graph=True)[0]
                l2_w_y = grad(boundary_pred, self.test_y_b, grad_outputs=torch.ones_like(boundary_pred),
                              create_graph=True, retain_graph=True)[0]

                # combining the partial derivatives
                w_xy = torch.hstack([l2_w_x, l2_w_y])

                # element wise dot product between the normal vectors and first partial derivatives
                gradient_n = torch.multiply(self.n_test, w_xy).sum(dim=1).unsqueeze(dim=1)

                # loss on the boundary
                boundary_residual_test = torch.multiply(self.alpha_test, gradient_n) + \
                                    torch.multiply(self.beta_test, boundary_pred) - self.g_test

                # storing losses in variables
                training_pde_residual = self.compute_linf(pde_residual_training)
                test_pde_residual = self.compute_linf(pde_residual_test)
                training_boundary_residual = self.compute_linf(boundary_residual_training)
                test_boundary_residual = self.compute_linf(boundary_residual_test)
                training_mse = self.compute_mse(training_pred, self.u_true)
                test_mse = self.compute_mse(test_pred, self.test_u_true)
                training_l2 = self.compute_l2(training_pred, self.u_true)
                test_l2 = self.compute_l2(test_pred, self.test_u_true)

                # logging losses and other important things in lists
                self.logged_results['training_losses'].append(loss_value)
                self.logged_results['training_mse_losses'].append(training_mse)
                self.logged_results['training_l2_losses'].append(training_l2)
                self.logged_results['training_pde_residual'].append(training_pde_residual)
                self.logged_results['training_boundary_residual'].append(training_boundary_residual)
                self.logged_results['test_mse_losses'].append(test_mse)
                self.logged_results['test_l2_losses'].append(test_l2)
                self.logged_results['test_pde_residual'].append(test_pde_residual)
                self.logged_results['test_boundary_residual'].append(test_boundary_residual)
                self.logged_results['epochs_list'].append(i)
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

                if i > 30 and (isnan(loss_value) or loss_value > 500):
                    # loss has exploded, this will trigger this run to restart
                    print(f"Loss exploded to: {loss_value}")
                    return False

                if i % self.print_interval == 0 and self.verbosity:
                    print('='*70)
                    print(f'Iter {i},\nTraining loss = {loss_value}')

                    # laplacian(pinn) - f
                    print('\nLaplacian(pinn) - f')
                    print('===TRAINING' + '='*36)
                    print(training_pde_residual)
                    print('===TEST' + '='*40)
                    print(test_pde_residual)

                    # alpha * d/dn pinn + beta * pinn - g
                    print('\nalpha * d/dn pinn + beta * pinn - g')
                    print('===TRAINING' + '='*36)
                    print(training_boundary_residual)
                    print('===TEST' + '='*40)
                    print(test_boundary_residual)

                    # mse errors
                    print('\nMSE errors w.r.t. u_true')
                    # training
                    print('===TRAINING' + '='*36)
                    print(f'PINN MSE: {training_mse}')
                    # test
                    print('===TEST' + '='*40)
                    print(f'PINN MSE: {test_mse}')

                    # l2 errors
                    print(f'\nL2 errors w.r.t. u_true')
                    # training
                    print('===TRAINING' + '='*36)
                    print(f'PINN L2: {training_l2}')
                    # test
                    print('===TEST' + '='*40)
                    print(f'PINN L2: {test_l2}')
            except KeyboardInterrupt:
                print('Keyboard Interrupt. Ending training.')
                return dict(self.logged_results)

        print(f"Learning rate annealed {annealing_counter} times")
        return dict(self.logged_results)

def save_results(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    precision_strings = ["float32"]
    orders = [2]
    # just one training set size
    size = 828
    alphas = [0.001, 0.0020, 0.0040, 0.0080, 0.0160, 0.0320, 0.0640, 0.1280, 0.2560]
    supervised_options = [False]
    activation_function = 'tanh'
    lr = 0.1

    for precision_string in precision_strings:
        PRECISION = torch.float32 if precision_string == "float32" else torch.float64
        results_folder = f"gpu_noisy_{precision_string}_synced_vanilla_results_noprint"
        for cur_supervised_option in supervised_options:
            print(f'\nGoing over supervised={cur_supervised_option}')
            for alpha_value in alphas:
                print(f"Going over alpha={alpha_value}")
                for order in orders:
                    supervised_file_bool = 'supervised' if cur_supervised_option else 'unsupervised'
                    file_name = f"{order}_{size}"
                    test_name = f"{order}_21748_test"
                    save_folder = f'../{results_folder}/vanilla_pinn/{order}/{size}/{alpha_value}/{supervised_file_bool}/'
                    if not os.path.isdir(save_folder):
                        os.makedirs(save_folder)
                    save_file_name = save_folder + 'results.json'

                    print('\n\n')
                    print('+'*70)
                    print(f'Running {precision_string} vanilla-PINN noisy on {file_name} with {activation_function}')
                    print('+'*70)
                    print('\n\n')

                    # read mat files
                    X_i = torch.tensor(loadmat(f"../scai/files_{file_name}/Xi.mat")["Xi"], dtype=PRECISION, requires_grad=True).to(device)
                    X_b = torch.tensor(loadmat(f"../scai/files_{file_name}/Xb.mat")["Xb"], dtype=PRECISION, requires_grad=True).to(device)
                    X_g = torch.tensor(loadmat(f"../scai/files_{file_name}/Xg.mat")["X_g"], dtype=PRECISION, requires_grad=True).to(device)
                    n = torch.tensor(loadmat(f"../scai/files_{file_name}/n.mat")["n"], dtype=PRECISION, requires_grad=True).to(device)
                    u_true = torch.tensor(loadmat(f"../scai/files_{file_name}/u.mat")["u"], dtype=PRECISION).to(device)
                    f = torch.tensor(loadmat(f"../scai/files_{file_name}/f_{alpha_value}.mat")["f"], dtype=PRECISION, requires_grad=True).to(device)
                    g = torch.tensor(loadmat(f"../scai/files_{file_name}/g_{alpha_value}.mat")["g"], dtype=PRECISION, requires_grad=True).to(device)
                    alpha = torch.tensor(loadmat(f"../scai/files_{file_name}/alpha.mat")["Neucoeff"], dtype=PRECISION, requires_grad=True).to(device)
                    beta = torch.tensor(loadmat(f"../scai/files_{file_name}/beta.mat")["Dircoeff"], dtype=PRECISION, requires_grad=True).to(device)
                    b_starts = X_i.shape[0]
                    b_end = b_starts + X_b.shape[0]


                    # test files
                    X_i_test = torch.tensor(loadmat(f"../scai/files_{test_name}/Xi.mat")["Xi"], dtype=PRECISION, requires_grad=True).to(device)
                    X_b_test = torch.tensor(loadmat(f"../scai/files_{test_name}/Xb.mat")["Xb"], dtype=PRECISION, requires_grad=True).to(device)
                    test_u_true = torch.tensor(loadmat(f"../scai/files_{test_name}/u.mat")["u"], dtype=PRECISION).to(device)
                    f_test = torch.tensor(loadmat(f"../scai/files_{test_name}/f.mat")["f"], dtype=PRECISION, requires_grad=True).to(device)
                    g_test = torch.tensor(loadmat(f"../scai/files_{test_name}/g.mat")["g"], dtype=PRECISION, requires_grad=True).to(device)
                    alpha_test = torch.tensor(loadmat(f"../scai/files_{test_name}/alpha.mat")["Neucoeff"], dtype=PRECISION, requires_grad=True).to(device)
                    beta_test = torch.tensor(loadmat(f"../scai/files_{test_name}/beta.mat")["Dircoeff"], dtype=PRECISION, requires_grad=True).to(device)
                    n_test = torch.tensor(loadmat(f"../scai/files_{test_name}/n.mat")["n"], dtype=PRECISION, requires_grad=True).to(device)

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
                        'precision': precision_string,
                        'activation': activation_function,
                        'order': 2, # activation order
                        'network_device': device_string,
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
                        'u_true': u_true,
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

