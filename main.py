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
parser.add_argument('--hunt', action='store_true', help="Hyperparameter search mode with orion")
parser.add_argument('--test', action='store_true', help="Train on train+val and test on test")
parser.add_argument('--eval_all', action='store_true')


def test(model: nn.Module, dataloader: data.DataLoader) -> float:
    print("Beginning testing")
    acc = 0.
    for i, (x, y) in enumerate(dataloader):
        pred = model(x)
        acc += (pred.max(-1)[1] == y).float().mean().item()
    acc /= i + 1
    return acc

def test_all(model: nn.Module, dataloader: data.DataLoader) -> float:
    print("Beginning testing")
    acc = torch.zeros(len(model.layers))
    for i, (x, y) in enumerate(dataloader):
        pred = model.forward_all(x)
        acc += (pred.max(-1)[1] == y[None,:]).float().mean(-1)
    acc /= i + 1
    return acc

def mnist_experiment(args):
    #60k samples
    dataset_for_test     = torchvision.datasets.MNIST('dataset', train=True , transform=torchvision.transforms.ToTensor(), download=True)
    #10k samples
    dataset_for_trainval = torchvision.datasets.MNIST('dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    if args.test:
        dataset_train = dataset_for_trainval
        dataset_test  = dataset_for_test
    else:
        n_train = 5000
        n_val = len(dataset_for_trainval) - n_train
        dataset_train, dataset_test = data.random_split(dataset_for_trainval, [n_train, n_val])
    print("Split for {}: {}/{} samples".format("test" if args.test else "validation", len(dataset_train), len(dataset_test)))
    layer1 = {
            'm':args.approx_m, 'd2':10, 'R':args.R, 'patch_dim':5, 'patch_stride':1, 'kernel':'rbf', 'avg_pooling_kernel_size':2, 'r':None, 'gamma':0.2,
            }
    layer2 = {
            'm':args.approx_m, 'd2':10, 'R':args.R, 'patch_dim':5, 'patch_stride':1, 'kernel':'rbf', 'avg_pooling_kernel_size':2, 'r':None, 'gamma':0.02,
            }
    model = layers.CCNN(img_shape=(1, 28, 28), layer_confs=[layer1, layer2])
    model.train(dataset_train, nn.CrossEntropyLoss(), 'fro', n_epochs=args.epochs, batch_size=64, lr=args.lr)
    dataloader_test = data.DataLoader(dataset_test, batch_size=64)
    if args.eval_all:
        acc = test_all(model, dataloader_test)
        for l, layer_acc in enumerate(acc):
            print("Accuracy: {:.2f}% on {} samples for layer {}".format(layer_acc*100, len(dataset_test), l))
        return acc[-1].item()
    else:
        acc = test(model, dataloader_test)
        print("Accuracy: {:.2f}% on {} samples".format(acc*100, len(dataset_test)))
        return acc

if __name__ == '__main__':
    args = parser.parse_args()
    acc = mnist_experiment(args)
    if args.hunt:
        assert not args.test
        orion.client.report_results([{
            'name': 'test_error_rate',
            'type': 'objective',
            'value': (1 - acc),
            }])
