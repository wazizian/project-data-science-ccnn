#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import layers
import argparse
import cnn
import light_ccnn
import logger

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.2)
parser.add_argument('--approx_m', type=int, default=100, help="m")
parser.add_argument('--R_projection', type=float, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--hunt', action='store_true', help="Hyperparameter search mode with orion")
parser.add_argument('--test', action='store_true', help="Evaluate on test set during training")
parser.add_argument('--eval_all', action='store_true', help="Test all the layers of the CCNN at the end")
parser.add_argument('--cnn', action='store_true', help="Switch from CCNN to CNN")
parser.add_argument('--activation', type=str, default="relu", help="Activation for CNN")
parser.add_argument('-v', '--verbose', action='store_true')

@torch.no_grad()
def test(model: nn.Module, dataloader: data.DataLoader) -> float:
    acc = 0.
    for i, (x, y) in enumerate(dataloader):
        pred = model(x)
        acc += (pred.max(-1)[1] == y).float().mean().item()
    acc /= i + 1
    return acc

@torch.no_grad()
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
        n_train = 5000
        n_out = len(dataset_for_trainval) - n_train
        dataset_train, _ = data.random_split(dataset_for_trainval, [n_train, n_out])
        n_out = len(dataset_for_test) - n_train
        dataset_test, _  = data.random_split(dataset_for_test, [n_train, n_out])
    else:
        n_train = 5000
        n_val = len(dataset_for_trainval) - n_train
        dataset_train, dataset_test = data.random_split(dataset_for_trainval, [n_train, n_val])
    dataloader_test = data.DataLoader(dataset_test, batch_size=64, num_workers=layers.NUM_WORKERS)
    if args.test:
        logger.Logger.dataloader_test = dataloader_test
    print("Split for {}: {}/{} samples".format("test" if args.test else "validation", len(dataset_train), len(dataset_test)))
    print("lr = {:.5f} gamma = {:.5f}".format(args.lr, args.gamma))
    layer1 = {
            'm':args.approx_m, 'd2':10, 'R':args.R_projection, 'patch_dim':5, 'patch_stride':1, 'kernel':'rbf', 'avg_pooling_kernel_size':2, 'r':16, 'gamma':args.gamma,
            }
    if args.cnn:
        model = cnn.CNN(img_shape=(1, 28, 28), layer_confs=[layer1], activation_func=args.activation)
    else:
        model = light_ccnn.LightCCNN((1, 28, 28), layer1)
    loggers = model.train(dataset_train, nn.CrossEntropyLoss(), 'fro', n_epochs=args.epochs, batch_size=64, lr=args.lr, verbose=args.verbose)
    if args.test:
        for i, log in enumerate(loggers):
            log.save('layer_{}'.format(i))
    elif args.eval_all:
        acc = test_all(model, dataloader_test)
        for l, layer_acc in enumerate(acc):
            print("Accuracy: {:.2f}% on {} samples for layer {}".format(layer_acc*100, len(dataset_test), l))
        if layers.SAFETY_CHECK:
            assert torch.norm(acc[-1] - test(model, dataloader_test)) <=1e-4
        return acc[-1].item()
    else:
        acc = test(model, dataloader_test)
        print("Accuracy: {:.2f}% on {} samples".format(acc*100, len(dataset_test)))
        return acc

if __name__ == '__main__':
    args = parser.parse_args()
    acc = mnist_experiment(args)
    if args.hunt:
        #Orion is needed at this point
        import orion
        import orion.client
        assert not args.test
        orion.client.report_results([{
            'name': 'test_error_rate',
            'type': 'objective',
            'value': (1 - acc),
            }])
