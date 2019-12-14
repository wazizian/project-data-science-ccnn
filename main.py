import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import layers

def test(model: nn.Module, dataloader: data.DataLoader) -> float:
    print("Beginning testing")
    acc = 0
    for i, (x, y) in enumerate(dataloader):
        pred = model(x)
        acc += (pred.max(-1)[1] == y).float().mean()
    acc /= i + 1
    return acc

def mnist_experiment():
    dataset = torchvision.datasets.MNIST('dataset', transform=torchvision.transforms.ToTensor(), download=True)
    dataset_train, dataset_test = data.random_split(dataset, [100, len(dataset) - 100])
    model = layers.CCNNLayer(m=100,
                            P=24*24, #number of patches
                            d2=10, #number of classes
                            R=1, patch_dim=5, patch_stride=1, kernel='rbf')
    model.train(dataset_train, nn.CrossEntropyLoss(), 'fro', n_epochs=10, batch_size=64)
    dataloader_test = data.DataLoader(dataset_test, batch_size=64)
    print("Test accuracy: {:.2f}% on {} samples".format(test(model, dataloader_test)*100, len(dataset_test)))

if __name__ == '__main__':
    mnist_experiment()

    
