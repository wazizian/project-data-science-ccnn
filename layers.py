import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import math
import logger
from typing import List

SAFETY_CHECK=False
NUM_WORKERS=4

class Conv2dRFF(nn.Module):
    """
    2d Convolutionnal layer with Random Fourier Features approximation

    It is the compossition of a 2d convolution and Random Fourier features
    approximation for the Gaussian kernel.

    It is inspired from sklearn.kernel_approximation.RBFSampler and implements
    the algorithm from "Random Features for Large-Scale Kernel Machines"
    by Ali Rahimi and Ben Recht.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
            gamma: float, **kwargs):
        super(Conv2dRFF, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.conv.weight.detach()
        self.conv.bias.detach()
        nn.init.normal_(self.conv.weight.data, std=math.sqrt(2*gamma))
        nn.init.uniform_(self.conv.bias.data, a=0., b=2*math.pi)
        self.activation = torch.cos
        self.scale = math.sqrt(2/out_channels)

    def forward(self, x):
        """
        x: shape b, h_in, w_in, in_channels
        returns: shape b, h_out, w_out, out_channels
        """
        x = self.conv(x)
        x = self.activation(x)
        return  self.scale * x



class CCNNLayerLinear(nn.Module):
    def __init__(self, m: int, h: int, d2: int, R: float, avg_pooling_kernel_size: int=1):
        """
        h: dim of new image
        m: rank of kernel factorization
        d2: output features
        R: norm constraint
        """
        super(CCNNLayerLinear, self).__init__()
        self.m = m
        self.P = h**2
        self.h = h
        self.hprime = self.h // avg_pooling_kernel_size
        self.Pprime = self.hprime**2
        self.d2 = d2
        self.R = R
        self.A = nn.Parameter(torch.zeros((self.m, self.hprime, self.hprime, self.d2)).normal_(), requires_grad=True)
        self.avg_pooling = nn.AvgPool2d(avg_pooling_kernel_size)

    def forward(self, z):
        assert z.size()[1:] == (self.m, self.h, self.h), (z.size(), self.m, self.h, self.h)
        z = self.avg_pooling(z)
        assert z.size()[1:] == (self.m, self.hprime, self.hprime), (z.size(), self.m, self.hprime, self.hprime)
        return torch.einsum('blij,ljio->bo', z, self.A)

    @torch.no_grad()
    def project(self, p):
        if p == 'fro':
            norm = torch.norm(self.A, p='fro')
            if norm > self.R:
                self.A.data = (self.R/norm) * self.A
        elif p == 'nuc':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def train(self, dataloader: data.DataLoader, criterion, p, n_epochs, lr, transform, verbose=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        print("Beginning training with {} batches of size <={}.".format(len(dataloader), dataloader.batch_size))
        log = logger.Logger(n_epochs, len(dataloader), verbose=verbose)
        for epoch in range(n_epochs):
            for iteration, (z, y) in enumerate(dataloader):
                output = self.forward(transform(z))
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.project(p)
                accuracy = (output.max(-1)[1] == y).float().mean()
                log.log_iteration(epoch, iteration, loss.item(), accuracy.item())
                log.log_test(epoch, iteration, self, transform)
        return log

    @torch.no_grad()
    def compute_approx_svd(self, r):
        U, s,V = torch.svd(self.A.data.view(self.m, -1), some=True)
        self.Uhat = U[:, :r]
        assert self.Uhat.size() == (self.m, r), (self.Uhat.size(), self.m, r)
        #For safety check
        if SAFETY_CHECK:
            assert r == self.m
            s = F.pad(s, pad=(0, V.size(1) - s.size(0)), value=0)
            self.other = torch.einsum('i,ij->ij', s, V.T)
            self.other = self.other.view(-1, self.hprime, self.hprime, self.d2)
            self.other = self.other[:r]

    @torch.no_grad()
    def apply_filters(self, Z_in: torch.Tensor) -> torch.Tensor:
        """
        Z has shape (b, m, h, h)
        self.Uhat has shape (m, r)
        output has shape (b, r, h', h')
        """
        assert Z_in.size()[1:] == (self.m, self.h, self.h)
        Z = self.avg_pooling(Z_in)
        assert    Z.size()[1:] == (self.m, self.hprime, self.hprime)
        Z = torch.einsum('blij,lf->bfij', Z, self.Uhat)
        assert    Z.size()[2:] == (self.hprime, self.hprime)
        #Safety check
        if SAFETY_CHECK:
            pred = torch.einsum('bfij,fjio->bo', Z, self.other)
            assert torch.norm(pred - self.forward(Z_in)) <= 1e-4, torch.norm(pred - self.forward(Z_in))
        return Z


class CCNNLayer(nn.Module):
    def __init__(self, img_shape, m, d2, R, patch_dim, patch_stride, kernel, r=None, gamma=0.2, avg_pooling_kernel_size:int =1):
        super(CCNNLayer, self).__init__()
        (c, h, w) = img_shape
        #this code is only meant for square images
        assert h == w
        self.P = ((h - patch_dim)//patch_stride + 1)**2
        self.h = (h - patch_dim)//patch_stride + 1
        self.convRFF = Conv2dRFF(c, m, patch_dim, gamma)
        self.linear = CCNNLayerLinear(m, self.h, d2, R, avg_pooling_kernel_size=avg_pooling_kernel_size)
        self.hprime = self.linear.hprime
        self.patch_dim = patch_dim
        self.patch_stride = patch_stride
        self.r = min(r,m) if r else m
        if r and self.r < r:
            print('Warning: r={} has been reduced to {}'.format(r, self.r))
        print('Creating CCNN layer for input of shape {}x{}x{} with m={} P={} P\'={} d2={}'.format(*img_shape, m, self.P, self.linear.Pprime, d2))    

    def train(self, dataset, criterion, p, n_epochs, batch_size, lr, transform=None, verbose=True) -> logger.Logger:
        if not transform:
            # x   has shape (b, c, h, h)
            # out has shape (b, m, self.h, self.h)
            transform = self.convRFF
        dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
        log = self.linear.train(dataloader, criterion, p, n_epochs, lr, transform, verbose=verbose)
        self.linear.compute_approx_svd(self.r)
        return log

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor, last=True) -> torch.Tensor:
        kernel_patches = self.convRFF(imgs)
        if last:
            return self.linear.forward(kernel_patches)
        else:
            # transformed has shape (b, m, h', h')
            transformed = self.linear.apply_filters(kernel_patches)
            # transformed has shape (b, r, h', h')
            return transformed

class CCNN(nn.Module):
    def __init__(self, img_shape, layer_confs):
        super(CCNN, self).__init__()
        in_shape = img_shape
        layer_lst = []
        for conf in layer_confs:
            current_layer = CCNNLayer(in_shape, **conf)
            layer_lst.append(current_layer)
            in_shape = (current_layer.r, current_layer.linear.hprime, current_layer.linear.hprime)
        self.layers = nn.ModuleList(layer_lst)

    def make_transform(self, layer_index):
        def transform(x: torch.Tensor) -> torch.Tensor:
            # x has shape (b, c, h, h)
            for i in range(layer_index):
                x = self.layers[i].forward(x, last=False)
            # x has shape (b, c, h, h) (where c is the last r, etc...)
            # out has shape (b, m, self.h, self.h)
            curr_layer = self.layers[layer_index]
            return curr_layer.convRFF(x)
        return transform


    def train(self, dataset, criterion, p, n_epochs, batch_size, lr, verbose=True) -> List[logger.Logger]:
        loggers = []
        for i, l in enumerate(self.layers):
            transform = self.make_transform(i)
            log = l.train(dataset, criterion, p, n_epochs, batch_size, lr, transform=transform, verbose=verbose)
            loggers.append(log)
        return loggers

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers[:-1]:
            x = l.forward(x, last=False)
        return self.layers[-1].forward(x, last=True)
    
    @torch.no_grad()
    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for l in self.layers[:-1]:
            predictions.append(l.forward(x, last=True))
            x = l.forward(x, last=False)
        predictions.append(self.layers[-1].forward(x, last=True))
        return torch.stack(predictions)

