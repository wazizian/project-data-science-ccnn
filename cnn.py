import torch
import torch.nn as nn
import torch.utils.data as data
import logger

"""
This class is designed to satisfy the same interface as CCNN
"""
class CNN(nn.Module):
    def __init__(self, img_shape, layer_confs, bias=False):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.linears = nn.ModuleList()
        in_channels = img_shape[0]
        h_in = img_shape[1]
        assert h_in == img_shape[2]
        for layer_conf in layer_confs:
            layer = nn.Sequential(
                    nn.Conv2d(in_channels, layer_conf['r'], kernel_size=layer_conf['patch_dim'], stride=layer_conf['patch_stride'], bias=bias),
                    nn.ReLU(),
                    nn.AvgPool2d(layer_conf['avg_pooling_kernel_size'])
                    )
            self.layers.append(layer)
            #same convention as CCNNLayer
            h = (h_in - layer_conf['patch_dim'])//layer_conf['patch_stride'] + 1
            hprime = h//layer_conf['avg_pooling_kernel_size']
            self.linears.append(nn.Linear(layer_conf['r'] * hprime * hprime, layer_conf['d2']))
            in_channels = layer_conf['r']
            h_in = hprime

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return self.linears[-1](torch.flatten(x, start_dim=1))

    @torch.no_grad()
    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        for conv,linear in zip(self.layers, self.linears):
            x = conv(x)
            predictions.append(linear(torch.flatten(x, start_dim=1)))
        return torch.stack(predictions)

    def train(self, dataset, criterion, _, n_epochs=100, batch_size=64, lr=1e-3, verbose=True):
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
                accuracy = (output.max(-1)[1] == y).float().mean()
                log.log_iteration(epoch, iteration, loss, accuracy)
        return logger



