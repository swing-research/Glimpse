"""
This script contains the vector model for the filter.
"""

import torch
import torch.nn as nn
from skimage.transform.radon_transform import _get_fourier_filter

class VectorModel(nn.Module):
    def __init__(self, init: str, size: int):
        super(VectorModel, self).__init__()
        self.size = size
        self.init = init


        if self.init == 'ones':
            self.vector = nn.Parameter(torch.ones(size))
        else:
            vector = _get_fourier_filter(size, init)
            self.vector = nn.Parameter(torch.tensor(vector, dtype=torch.float32))

    def forward(self, x: int):
        return self.vector
