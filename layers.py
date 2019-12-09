import torch
import torch.nn as nn
from kernels import ApproxKernelMachine
import numpy as np
import sklearn.feature_extraction.image as image

class CCNNLayerLinear(nn.Module):
    def __init__(self, m: int, P: int, d2: int, R: float):
        """
        P: nb of patches
        m: rank of kernel factorization
        d2: output features
        R: norm constraint
        """
        super(CCNNLayerLinear, self).__init__()
        self.A = nn.Linear(m * P, d2, bias=False)
        self.m = m
        self.P = P
        self.d2 = d2
        self.R = R

    def forward(self, z):
        assert z.size()[1:] == (self.P, self.m)
        return self.A(z.view(-1, self.P * self.m))

    @torch.no_grad()
    def project(self, p):
        if p == 'fro':
            norm = torch.norm(self.A.weight, p='fro')
            if norm > self.R:
                self.A = (self.R/norm) * self.A
        elif p == 'nuc':
            raise ValueError("Nuclear norm not implemented yet")
        else:
            raise ValueError("Unsupported norm")


class CCNNLayer(nn.Module):
    def __init__(self, m, P, d2, R, patch_dim, patch_stride, kernel, gamma=0.2):
        super(CCNNLayer, self).__init__()
        self.kernel = ApproxKernelMachine(kernel, m, gamma=gamma)
        self.linear = CCNNLayerLinear(m, P, d2, R)
        self.patch_dim = patch_dim
        self.patch_stride = patch_size
    
    #WARNING: the rest is maybe not made for multichannels
    def extract_patches(self, imgs: torch.Tensor):
        """
        imgs has shape (b, c, h, w)
        output has shape (n_samples, P, d1) where d1 = self.patch_dim**2
        """
        (b, c, h, w) = imgs.size()
        patches = imgs.unfold(2, self.patch_dim, self.patch_stride).unfold(3, self.patch_dim, self.patch_stride)
        patches = patches.view(b, self.P, self.patch_dim**2)
        return patches

