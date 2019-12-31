#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import layers
import argparse
import orion
import orion.client

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--approx_m', type=int, default=100, help="m")
parser.add_argument('--R', type=float, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--hunt', type=bool, action='store_true', help="Hyperparameter search mode with orion")


def test(model: nn.Module, dataloader: data.DataLoader) -> float:
    print("Beginning testing")
    acc = 0.
    for i, (x, y) in enumerate(dataloader):
        pred = model(x)
        acc += (pred.max(-1)[1] == y).float().mean().item()
    acc /= i + 1
    return acc

def mnist_experiment(args):
    dataset = torchvision.datasets.MNIST('dataset', transform=torchvision.transforms.ToTensor(), download=True)
    n_train = 5000
    dataset_train, dataset_test = data.random_split(dataset, [n_train, len(dataset) - n_train - 100])
    model = layers.CCNNLayer(m=args.approx_m,
                            img_shape=(1, 28, 28),
                            d2=10, #number of classes
                            R=args.R, patch_dim=5, patch_stride=1, kernel='rbf')
    model.train(dataset_train, nn.CrossEntropyLoss(), 'fro', n_epochs=args.epochs, batch_size=64, lr=args.lr)
    dataloader_test = data.DataLoader(dataset_test, batch_size=64)
    acc = test(model, dataloader_test)
    print("Test accuracy: {:.2f}% on {} samples".format(acc*100, len(dataset_test)))
    return acc

if __name__ == '__main__':
    args = parser.parse_args()
    acc = mnist_experiment(args)
    if arsg.hunt:
        orion.client.report_results([{
            'name': 'test_error_rate',
            'type': 'objective',
            'value': (1 - acc),
            }])

    
