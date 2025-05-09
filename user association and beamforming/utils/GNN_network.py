import numpy as np
import torch
import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import torch
from torch_geometric.data import Dataset, Data
import torch
import torch.nn as nn

class Beampredict_fully_NN(nn.Module):
    def __init__(self, input_size, hidden_sizes, Batch_normal, output_size, num_groups=32, dropout_prob=0.0):
        super(Beampredict_fully_NN, self).__init__()
        self.layers = nn.ModuleList()  # Stores all the linear layers
        self.norms = nn.ModuleList()  # Stores all the normalization layers
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()
        previous_size = input_size

        # Initialize layers dynamically based on hidden_sizes
        for i, size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(previous_size, size))
            if Batch_normal == 'GN':
                self.norms.append(nn.GroupNorm(num_groups=min(num_groups, size), num_channels=size))
            elif Batch_normal == 'LN':
                self.norms.append(nn.LayerNorm(size))
            elif Batch_normal == 'BN':
                self.norms.append(nn.BatchNorm1d(size))
            elif Batch_normal == 'None':
                self.norms.append(nn.Identity())
            previous_size = size

        # Output layer without normalization
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = self.relu(norm(layer(x)))
            if self.dropout.p > 0:
                x = self.dropout(x)

        x = self.output_layer(x)  # Apply output layer without activation and normalization
        return x
