class Logger:
    def __init__(self, n_epochs, n_iterations, print_freq=None):
        self.losses = []
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.print_freq = print_freq if print_freq else n_iterations//100

    def log_iteration(self, epoch, iteration, loss):
        self.losses.append(loss)
        if epoch % self.print_freq == 0:
            print("[{}/{}, {}/{}] Loss = {}".format(epoch, self.n_epochs, iteration, self.n_iterations, loss))
