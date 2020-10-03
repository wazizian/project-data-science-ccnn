import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import logger
import layers

class LightCCNN(nn.Module):
    def __init__(self, img_shape, layer_conf, bias=False, activation_func="relu"):
        super(LightCCNN, self).__init__()
        in_channels = img_shape[0]
        h_in = img_shape[1]
        assert h_in == img_shape[2]
        self.layer = nn.Sequential(
                layers.Conv2dRFF(in_channels, layer_conf['r'], kernel_size=layer_conf['patch_dim'], gamma=layer_conf['gamma'], stride=layer_conf['patch_stride']),
                nn.AvgPool2d(layer_conf['avg_pooling_kernel_size'])
                )
        h = (h_in - layer_conf['patch_dim'])//layer_conf['patch_stride'] + 1
        hprime = h//layer_conf['avg_pooling_kernel_size']
        self.linear = nn.Linear(layer_conf['r'] * hprime * hprime, layer_conf['d2'])
        self.R = layer_conf['R']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return self.linear(torch.flatten(x, start_dim=1))
    
    @torch.no_grad()
    def project(self, p):
        if p == 'fro':
            norm = torch.norm(self.linear.weight, p='fro')
            if norm > self.R:
                self.linear.weight.data = (self.R/norm) * self.linear.weight
        elif p == 'nuc':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def train(self, dataset, criterion, p, n_epochs=100, batch_size=64, lr=1e-3, verbose=True):
        dataloader = data.DataLoader(dataset, batch_size=64)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        print("Beginning training with {} batches of size <={}.".format(len(dataloader), dataloader.batch_size))
        log = logger.Logger(n_epochs, len(dataloader), verbose=verbose)
        for epoch in range(n_epochs):
            for iteration, (x, y) in enumerate(dataloader):
                output = self.forward(x)
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.project(p)
                accuracy = (output.max(-1)[1] == y).float().mean()
                log.log_iteration(epoch, iteration, loss.item(), accuracy.item())
                log.log_test(epoch, iteration, self, lambda x: x)
        return log


