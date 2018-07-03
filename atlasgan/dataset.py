"""
This module defines the ATLAS RPV images datasets.
"""

# System imports
from __future__ import division

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

# TODO: add the theory mass parameters for conditioning
class RPVImages(Dataset):
    """Dataset wrapping RPV image tensors."""
    def __init__(self, input_file, n_samples=None, scale=None):
        # Load the data
        with np.load(input_file) as f:
            fdata = f['hist']
            if n_samples is not None and n_samples > 0:
                self.data = torch.from_numpy(fdata[:n_samples, None].astype(np.float32))
            else:
                self.data = torch.from_numpy(fdata[:, None]).astype(np.float32)
        if scale is not None:
            self.data = transform_data(self.data, scale)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.data.size(0)
