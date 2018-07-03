"""
This module defines the ATLAS RPV images datasets.
"""

# System imports
from __future__ import division
import os

# External imports
import numpy as np
import torch
from torch.utils.data import Dataset

def transform_data(x, scale):
    """Standard transform of the data for the models"""
    return x / scale

def inverse_transform_data(x, scale, threshold=0):
    """Undo the standard transform; shouldn't need threshold anymore"""
    x = x * scale
    x[x < threshold] = 0
    return x

def generate_noise(n_samples, noise_dim):
    """Generate the random noise tensor for the GAN"""
    return torch.FloatTensor(n_samples, noise_dim, 1, 1).normal_(0, 1)

class RPVImages(Dataset):
    """Dataset wrapping RPV image tensors."""
    def __init__(self, input_file, n_samples=None, scale=4e6):
        # Load the data
        with np.load(input_file) as f:
            fdata = f['hist']
            if n_samples is not None:
                fdata = fdata[:n_samples]
            # Add channels dimension and convert to PyTorch float32
            data = torch.from_numpy(fdata[:, None].astype(np.float32))
        # Apply standard data transformation
        self.data = transform_data(data, scale)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.data.size(0)

# TODO: add control to set number of mass params.
class RPVCondImages(Dataset):
    """Dataset wrapping RPV image tensors and theory mass parameters"""
    def __init__(self, input_file, n_samples=None, scale=4e6):
        # Load the data
        with np.load(input_file) as f:
            fdata = f['hist']
            if n_samples is not None:
                fdata = fdata[:n_samples]
            # Add channels dimension and convert to PyTorch float32
            data = torch.from_numpy(fdata[:, None].astype(np.float32))
        # Apply standard data transformation
        self.data = transform_data(data, scale)
        # Infer the mass parameters from the file name
        _, mglu, mneu = os.path.basename(input_file).split('_')[:3]
        # Fixed transformations
        mglu = (float(mglu) - 1400) / 500.
        mneu = (float(mneu) - 250) / 1400.
        self.cond = torch.FloatTensor([mglu, mneu])

    def __getitem__(self, index):
        return self.data[index], self.cond

    def __len__(self):
        return self.data.size(0)
