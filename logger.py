class Logger:
    def __init__(self, n_epochs, n_iterations, print_freq=None, verbose=True):
        self.losses = []
        self.accuracies = []
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.print_freq = print_freq if print_freq else 1
        self.verbose = verbose

    def log_iteration(self, epoch, iteration, loss, accuracy):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        if iteration % self.print_freq == 0 and self.verbose:
            print("[{}/{}, {:2d}/{}] Loss = {:.2f} Accuracy = {:.2f}%".format(epoch, self.n_epochs, iteration, self.n_iterations, loss, accuracy*100))
