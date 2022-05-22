from collections import OrderedDict
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import os

class W(nn.Module):
    def __init__(self, config={}):
        torch.manual_seed(0)

        super().__init__()

        assert config != {} or config is not None, "Config cannot be empty"

        self.config = config
        print(f"Using precision = {self.config['precision']}")

        if self.config['precision'] == "float32":
            self.precision = torch.float32
        elif self.config['precision'] == "float64":
            self.precision = torch.float64

        if self.config['activation'] == 'tanh':
            self.activation = nn.Tanh()
            self.activation_name = str(self.activation)
        elif self.config['activation'] == 'repu':
            self.activation = RePU(order=self.config['order'])
            if self.config['order'] == 2:
                self.activation_name = 'requ'
            elif self.config['order'] == 3:
                self.activation_name = 'recu'

        self.layers = self.config['layers']
        node_per_layer = self.config['nodes']
        hidden_layers = [node_per_layer for _ in range(self.config['layers'])]
        self.layers = [self.config['spatial_dim']] + hidden_layers + [1]

        self.layer_list = OrderedDict()

        for i in range(len(self.layers) - 1):
            if i < len(self.layers) - 2:
                self.layer_list.update({
                    f'linear_{i+1}': nn.Linear(in_features=self.layers[i], out_features=self.layers[i + 1]),
                    f'activation_{self.activation_name}_{i+1}': self.activation
                })
            else:
                self.layer_list.update({
                    f'linear_{i+1}': nn.Linear(in_features=self.layers[i], out_features=self.layers[i + 1])
                })

        self.net = nn.Sequential(self.layer_list)

        for param in self.net.parameters():
            param.requires_grad = True

        if self.config.get('network_device') is not None:
            if self.config['network_device'] == 'cpu':
                self.device = torch.device('cpu')
            elif self.config['network_device'] == 'cuda':
                self.device = torch.device('cuda')
        else:
            # CUDA support
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                torch.cuda.manual_seed_all(0)
            else:
                self.device = torch.device('cpu')

        self.net = self.net.to(self.precision).to(self.device)

        for param in self.net.parameters():
            assert param.dtype == self.precision
        print(f'All parameter data type are: {self.precision}')

    def forward(self, features):
        features = features.to(self.precision).to(self.device)

        output = self.net(features)

        return output
