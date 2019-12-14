import torch
import torch.nn as nn
import torch.utils.data as data
from kernels import ApproxKernelMachine
import logger
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
                self.A.weight.data = (self.R/norm) * self.A.weight
        elif p == 'nuc':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def train(self, dataloaderZ: data.DataLoader, criterion, p, n_epochs):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
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



class CCNNLayer(nn.Module):
    def __init__(self, m, P, d2, R, patch_dim, patch_stride, kernel, gamma=0.2):
        super(CCNNLayer, self).__init__()
        self.kernel = ApproxKernelMachine(kernel, m, gamma=gamma)
        self.linear = CCNNLayerLinear(m, P, d2, R)
        self.patch_dim = patch_dim
        self.patch_stride = patch_stride
        self.P = P
    
    #WARNING: the rest is maybe not made for multichannels
    def extract_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs has shape (b, c, h, w)
        output has shape (n_samples, P, d1) where d1 = self.patch_dim**2
        """
        (b, c, h, w) = imgs.size()
        patches = imgs.unfold(2, self.patch_dim, self.patch_stride).unfold(3, self.patch_dim, self.patch_stride)
        patches = patches.reshape(b, self.P, self.patch_dim**2)
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

    def train(self, dataset, criterion, p, n_epochs, batch_size) -> logger.Logger:
        train_dataset = self.build_train_dataset(dataset)
        dataloader = data.DataLoader(train_dataset, batch_size=batch_size)
        return self.linear.train(dataloader, criterion, p, n_epochs)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patches(imgs)
        kernel_patches = self.kernel.transform(patches.numpy())
        return self.linear.forward(kernel_patches)


