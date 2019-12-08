import torch
import torch.nn as nn

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






