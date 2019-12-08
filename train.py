import torch
import torch.nn
import layers
import logger

def optimizeCCNNLinear(model: layers.CCNNLayerLinear, dataloaderZ: torch.utils.data.DataLoader, criterion, p, n_epochs):
    optimizer = torch.optim.SGD(model.parameters())
    log = logger.Logger(n_epochs, len(dataloaderZ))
    for epoch in range(n_epochs):
        for iteration, (z, y) in enumerate(dataloaderZ):
            output = model.forward(z)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.project(p)
            log.log_iteration(epoch, iteration, loss)
    return logger


            

