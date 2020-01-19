import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from kernels import LightApproxKernelMachine
import logger
import numpy as np
import sklearn.feature_extraction.image as image
from typing import List

SAFETY_CHECK=False

class CCNNLayerLinear(nn.Module):
    def __init__(self, m: int, P: int, d2: int, R: float, avg_pooling_kernel_size: int=1):
        """
        P: nb of patches
        m: rank of kernel factorization
        d2: output features
        R: norm constraint
        """
        super(CCNNLayerLinear, self).__init__()
        self.m = m
        self.P = P
        self.h = int(np.sqrt(self.P))
        assert self.h**2 == self.P
        self.hprime = self.h // avg_pooling_kernel_size
        self.Pprime = self.hprime**2
        self.d2 = d2
        self.R = R
        self.A = nn.Parameter(torch.zeros((self.m, self.hprime, self.hprime, self.d2)).normal_(), requires_grad=True)
        self.avg_pooling_layer = nn.AvgPool2d(avg_pooling_kernel_size)

    def avg_pooling(self, z: torch.Tensor) -> torch.Tensor:
        """
        z has size (b, P, d1) and we assume P = h**2
        we apply 2D average pooling to z.view(b, h, h, d1)
        """
        b, _, _, d1 = z.size()
        assert (z.size(1) == self.h) and (z.size(2) == self.h)
        return self.avg_pooling_layer(z.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


    def forward(self, z):
        assert z.size()[1:] == (self.h, self.h, self.m), (z.size(), self.h, self.h, self.m)
        z = self.avg_pooling(z)
        assert z.size()[1:] == (self.hprime, self.hprime, self.m), (z.size(), self.hprime, self.hprime, self.m)
        return torch.einsum('bijl,ljio->bo', z, self.A)

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

    def train(self, dataloader: data.DataLoader, criterion, p, n_epochs, lr, transform):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        print("Beginning training with {} batches of size <={}.".format(len(dataloader), dataloader.batch_size))
        log = logger.Logger(n_epochs, len(dataloader))
        for epoch in range(n_epochs):
            for iteration, (z, y) in enumerate(dataloader):
                output = self.forward(transform(z))
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.project(p)
                accuracy = (output.max(-1)[1] == y).float().mean()
                log.log_iteration(epoch, iteration, loss, accuracy)
        return logger

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
        Z has shape (b, h, h, m)
        self.Uhat has shape (m, r)
        output has shape (b, h', h', r)
        """
        Z = self.avg_pooling(Z_in)
        Z = torch.einsum('bijl,lf->bijf', Z, self.Uhat)
        #Safety check
        if SAFETY_CHECK:
            pred = torch.einsum('bijf,fjio->bo', Z, self.other)
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
        self.d1 = c * patch_dim**2
        self.kernel = LightApproxKernelMachine(kernel, self.d1, m, gamma=gamma)
        self.linear = CCNNLayerLinear(m, self.P, d2, R, avg_pooling_kernel_size=avg_pooling_kernel_size)
        self.hprime = self.linear.hprime
        self.patch_dim = patch_dim
        self.patch_stride = patch_stride
        self.r = min(r,m) if r else m
        if r and self.r < r:
            print('Warning: r={} has been reduced to {}'.format(r, self.r))
        print('Creating CCNN layer for input of shape {}x{}x{} with m={} P={} P\'={} d2={}'.format(*img_shape, m, self.P, self.linear.Pprime, d2))
    
    def extract_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs has shape (b, c, h, h)
        output has shape (b, self.h, self.h, self.d1) 
        """
        (b, c, h, w) = imgs.size()
        assert h == w
        patches = imgs.unfold(2, self.patch_dim, self.patch_stride).unfold(3, self.patch_dim, self.patch_stride)
        assert patches.size() == (b, c, self.h, self.h, self.patch_dim, self.patch_dim)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(b, self.h, self.h, c * self.patch_dim ** 2)
        return patches

    def build_patch_dataset(self, dataset: data.Dataset) -> (np.ndarray, torch.Tensor):
        length = len(dataset)
        (inputs, labels) = next(data.DataLoader(dataset, batch_size=length).__iter__())
        inputs = self.extract_patches(inputs)
        return inputs.numpy(), labels

    def build_train_dataset(self, dataset: data.Dataset) -> data.Dataset:
        inputs, labels = self.build_patch_dataset(dataset)
        self.kernel.fit(inputs)
        return self.kernel.buid_kernel_patch_dataset(labels)

    def train(self, dataset, criterion, p, n_epochs, batch_size, lr, transform) -> logger.Logger:
        if not transform:
            # x   has shape (b, c, h, h)
            # out has shape (b, self.h, self.h, self.d1)
            transform = lambda x: self.kernel.transform(self.extract_patches(x))
        dataloader = data.DataLoader(dataset, batch_size=batch_size)
        logger = self.linear.train(dataloader, criterion, p, n_epochs, lr, transform)
        self.linear.compute_approx_svd(self.r)
        return logger

    def build_next_layer_dataset(self) -> data.Dataset:
        (inputs, labels) = next(data.DataLoader(self.train_dataset, batch_size=len(self.train_dataset)).__iter__())
        transformed = self.linear.apply_filters(inputs)
        # transformed is supposed to have shape (b, h', h', r)
        assert transformed.size()[1:] == (self.hprime, self.hprime, self.r)
        # r is now the number of channels
        transformed = transformed.permute(0, 3, 1, 2)
        return data.TensorDataset(transformed, labels)

    def forward(self, imgs: torch.Tensor, last=True) -> torch.Tensor:
        patches = self.extract_patches(imgs)
        kernel_patches = self.kernel.transform(patches)
        if last:
            return self.linear.forward(kernel_patches)
        else:
            # transformed has shape (b, h', h', m)
            transformed = self.linear.apply_filters(kernel_patches)
            # transformed has shape (b, h', h', r)
            transformed = transformed.permute(0, 3, 1, 2)
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
            # out has shape (b, self.h, self.h, self.d1)
            curr_layer = self.layers[layer_index]
            return curr_layer.kernel.transform(curr_layer.extract_patches(x))
        return transform


    def train(self, dataset, criterion, p, n_epochs, batch_size, lr) -> List[logger.Logger]:
        loggers = []
        for i, l in enumerate(self.layers):
            transform = self.make_transform(i)
            logger = l.train(dataset, criterion, p, n_epochs, batch_size, lr, transform=transform)
            loggers.append(logger)
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

