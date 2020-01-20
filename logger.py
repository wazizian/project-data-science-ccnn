import csv
import torch

class Logger:
    # used for computing test accuracy during training
    # should have been handled better
    # by for instance complexifying this class
    dataloader_test = None

    def __init__(self, n_epochs, n_iterations, print_freq=None, verbose=True):
        self.losses = []
        self.accuracies = []
        self.test_accuracies = []
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.print_freq = print_freq if print_freq else 1
        self.test_freq = 5
        self.verbose = verbose

    def log_iteration(self, epoch, iteration, loss, accuracy):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        if iteration % self.print_freq == 0 and self.verbose:
            print("[{}/{}, {:2d}/{}] Loss = {:.2f} Accuracy = {:.2f}%".format(epoch, self.n_epochs, iteration, self.n_iterations, loss, accuracy*100))
    
    @torch.no_grad()
    def log_test(self, epoch, iteration, model, transform):
        if not self.dataloader_test or iteration % self.test_freq != 0:
            return
        acc = 0.
        for i, (x, y) in enumerate(self.dataloader_test):
            pred = model(transform(x))
            acc += (pred.max(-1)[1] == y).float().mean().item()
        acc /= i + 1
        self.test_accuracies.append(acc)

    def save(self, name):
        with open(name + '_train_loss.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerows([[loss] for loss in self.losses])
        with open(name + '_test_acc.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerows([[acc] for acc in self.test_accuracies])



