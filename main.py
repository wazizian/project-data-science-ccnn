import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import layers

def test(model: nn.Module, dataloader: data.DataLoader) -> float:
    acc = 0
    for i, (x, y) in enumerate(dataloader):
        pred = model(x)
        acc += (pred.max(-1)[1] == y).mean()
    acc /= i + 1
    return acc

def mnist_experiment():
    dataset_train = torchvision.datasets.MNIST('dataset', train=True,  transform=torchvision.transforms.ToTensor(), download=True)
    dataset_test  = torchvision.datasets.MNIST('dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)
    model = layers.CCNNLayer(m=100,
                            P=24*24, #number of patches
                            d2=10, #number of classes
                            R=1, patch_dim=5, patch_stride=1, kernel='rbf')
    model.train(dataset_train, nn.CrossEntropyLoss, 'fro', 1000, 64)
    dataloader_test = data.DataLoader(dataset_test, batch_size=64)
    print("Test accuracy: {}%".format(test(model, dataloader_test)))

if __name__ == '__main__':
    mnist_experiment()

    
