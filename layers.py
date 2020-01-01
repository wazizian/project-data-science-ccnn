import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from kernels import ApproxKernelMachine
import logger
import numpy as np
import sklearn.feature_extraction.image as image
from typing import List

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
        self.A = nn.Linear(m * self.Pprime, d2, bias=False)
        self.d2 = d2
        self.R = R
        self.avg_pooling_layer = nn.AvgPool2d(avg_pooling_kernel_size)

    def avg_pooling(self, z: torch.Tensor) -> torch.Tensor:
        """
        z has size (b, P, d1) and we assume P = h**2
        we apply 2D average pooling to z.view(b, h, h, d1)
        """
        b, P, d1 = z.size()
        assert P == self.P
        return self.avg_pooling_layer(z.view(b, self.h, self.h, d1)).view(b, -1, d1)


    def forward(self, z):
        assert z.size()[1:] == (self.P, self.m)
        z = self.avg_pooling(z)
        assert z.size()[1:] == (self.Pprime, self.m)
        return self.A(z.view(-1, self.Pprime * self.m))

    @torch.no_grad()
    def project(self, p):
        if p == 'fro':
            norm = torch.norm(self.A.weight, p='fro')
            if norm > self.R:
                self.A.weight.data = (self.R/norm) * self.A.weight
        elif p == 'nuc':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def train(self, dataloaderZ: data.DataLoader, criterion, p, n_epochs, lr):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        print("Beginning training with {} batches of size <={}.".format(len(dataloaderZ), dataloaderZ.batch_size))
        log = logger.Logger(n_epochs, len(dataloaderZ))
        for epoch in range(n_epochs):
            for iteration, (z, y) in enumerate(dataloaderZ):
                output = self.forward(z)
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
        U, _, _ = torch.svd(self.A.weight.T.reshape(self.m, -1), some=False)
        self.Uhat = U[:, :r]
        assert self.Uhat.size() == (self.m, r), (self.Uhat.size(), self.m, r)

    @torch.no_grad()
    def apply_filters(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z has shape (b, P, m)
        self.Uhat has shape (m, r)
        """
        Z = self.avg_pooling(Z)
        return F.linear(Z, self.Uhat.T).permute(0, 2, 1)


class CCNNLayer(nn.Module):
    def __init__(self, img_shape, m, d2, R, patch_dim, patch_stride, kernel, r=None, gamma=0.2, avg_pooling_kernel_size:int =1):
        super(CCNNLayer, self).__init__()
        (c, h, w) = img_shape
        #this code is only meant for square images
        assert h == w
        self.P = ((h - patch_dim)//patch_stride + 1)**2
        self.kernel = ApproxKernelMachine(kernel, m, gamma=gamma)
        self.linear = CCNNLayerLinear(m, self.P, d2, R, avg_pooling_kernel_size=avg_pooling_kernel_size)
        self.patch_dim = patch_dim
        self.patch_stride = patch_stride
        #self.r = min(*self.linear.A.weight.T.reshape(m, -1).size(), r) if r else min(*self.linear.A.weight.size())
        self.r = min(r,m) if r else m
        if r and self.r < r:
            print('Warning: r={} has been reduced to {}'.format(r, self.r))
        print('Creating CCNN layer for input of shape {}x{}x{} with m={} P={} P\'={} d2={}'.format(*img_shape, m, self.P, self.linear.Pprime, d2))
    
    def extract_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs has shape (b, c, h, w)
        output has shape (n_samples, P, d1) where d1 = c * self.patch_dim**2
        """
        (b, c, h, w) = imgs.size()
        patches = imgs.unfold(2, self.patch_dim, self.patch_stride).unfold(3, self.patch_dim, self.patch_stride)
        patches = patches.reshape(b, self.P, c * self.patch_dim**2)
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

    def train(self, dataset, criterion, p, n_epochs, batch_size, lr) -> logger.Logger:
        self.train_dataset = self.build_train_dataset(dataset)
        dataloader = data.DataLoader(self.train_dataset, batch_size=batch_size)
        logger = self.linear.train(dataloader, criterion, p, n_epochs, lr)
        self.linear.compute_approx_svd(self.r)
        return logger

    def build_next_layer_dataset(self) -> data.Dataset:
        (inputs, labels) = next(data.DataLoader(self.train_dataset, batch_size=len(self.train_dataset)).__iter__())
        transformed = self.linear.apply_filters(inputs)
        # transformed is supposed to have shape (b, r, Pprime)
        # r is now the number of channels
        # as Pprime = hprime**2, we can see transformed as (b, r, hprime, hprime) dataset of images
        assert transformed.size(1) == self.r, (transformed.size(), self.r)
        assert transformed.size(2) == self.linear.Pprime
        transformed = transformed.view(-1, self.r, self.linear.hprime, self.linear.hprime)
        return data.TensorDataset(transformed, labels)

    def forward(self, imgs: torch.Tensor, last=True) -> torch.Tensor:
        patches = self.extract_patches(imgs)
        kernel_patches = self.kernel.transform(patches.numpy()) #has shape b, P, m
        if last:
            return self.linear.forward(kernel_patches)
        else:
            transformed = self.linear.apply_filters(kernel_patches)
            transformed = transformed.view(-1, self.r, self.linear.hprime, self.linear.hprime)
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

    def train(self, dataset, criterion, p, n_epochs, batch_size, lr) -> List[logger.Logger]:
        train_dataset = dataset
        loggers = []
        for l in self.layers:
            logger = l.train(train_dataset, criterion, p, n_epochs, batch_size, lr)
            loggers.append(logger)
            train_dataset = l.build_next_layer_dataset()
        return loggers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers[:-1]:
            x = l.forward(x, last=False)
        return self.layers[-1].forward(x, last=True)
    
    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for l in self.layers[:-1]:
            predictions.append(l.forward(x, last=True))
            x = l.forward(x, last=False)
        predictions.append(self.layers[-1].forward(x, last=True))
        return torch.stack(predictions)

