import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from kernels import ApproxKernelMachine
import logger
import numpy as np
import sklearn.feature_extraction.image as image

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
        U, _, _ = torch.svd(self.A.weight)
        self.Uhat = U[:, :r]

    @torch.no_grad()
    def apply_filters(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z has shape (b, P, m)
        self.Uhat has shape (m, r)
        """
        Z = self.avg_pooling(Z)
        return F.linear(Z, self.Uhat.T).permute(1, 2)


class CCNNLayer(nn.Module):
    def __init__(self, m, img_shape, d2, R, patch_dim, patch_stride, kernel, gamma=0.2, first=True, last=True, avg_pooling_kernel_size:int =1):
        super(CCNNLayer, self).__init__()
        (c, h, w) = img_shape
        #this code is only meant for square images
        assert h == w
        self.P = ((h - patch_dim)//patch_stride + 1)**2
        self.kernel = ApproxKernelMachine(kernel, m, gamma=gamma)
        self.linear = CCNNLayerLinear(m, self.P, d2, R, avg_pooling_kernel_size=avg_pooling_kernel_size)
        self.patch_dim = patch_dim
        self.patch_stride = patch_stride
        self.first, self.last = first, last
        print('Creating CCNN layer for input of shape {}x{}x{} with m={} P={}'.format(*img_shape, m, self.P))
    
    #WARNING: the rest is maybe not made for multichannels
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
        if self.first:
            inputs = self.extract_patches(inputs)
        return inputs.numpy(), labels

    def build_train_dataset(self, dataset: data.Dataset) -> data.Dataset:
        inputs, labels = self.build_patch_dataset(dataset)
        self.kernel.fit(inputs)
        return self.kernel.buid_kernel_patch_dataset(labels)

    def train(self, dataset, criterion, p, n_epochs, batch_size, lr) -> logger.Logger:
        if self.first:
            self.train_dataset = self.build_train_dataset(dataset)
        else:
            self.train_dataset = dataset
        dataloader = data.DataLoader(self.train_dataset, batch_size=batch_size)
        return self.linear.train(dataloader, criterion, p, n_epochs, lr)

    def build_next_layer_dataset(self) -> data.Dataset:
        (inputs, labels) = next(data.DataLoader(self.train_dataset, batch_size=len(self.train_dataset).__iter__()))
        transformed = self.linear.apply_filters(inputs)
        return data.TensorDataset(transformed, labels)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patches(imgs) if self.first else imgs
        kernel_patches = self.kernel.transform(patches.numpy()) #has shape b, P, m
        if self.last:
            return self.linear.forward(kernel_patches)
        else:
            return self.linear.apply_filters(kernel_patches)

#class CCNN(nn.Module):
#    def __init__(self, m_lst, P_lst, d_lst, R, patch_dim_lst, patch_stride_lst, kernel_lst, gamma_lst):

