"""
This module defines the ATLAS RPV images datasets.
"""

# External imports
import numpy as np
import torch
from torch.utils.data import Dataset

# TODO: add the theory mass parameters for conditioning
class RPVImages(Dataset):
    """Dataset wrapping RPV image tensors."""
    def __init__(self, input_file, n_samples=None, from_back=False):
        # Load the data
        with np.load(input_file) as f:
            fdata = f['hist']
            if n_samples is not None and n_samples > 0:
                self.data = torch.from_numpy(fdata[:n_samples, None].astype(np.float32))
            else:
                self.data = torch.from_numpy(fdata[:, None]).astype(np.float32)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.data.size(0)
