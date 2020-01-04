import numpy as np
import sklearn.kernel_approximation as kernel_approximation
import torch
import torch.utils.data as data

class ApproxKernelMachine:
    def __init__(self, kernel, m, gamma=0.2):
        """
        kernel(str): either 'rbf' or 'polynomial'
        """
        self.m = m
        if kernel == 'rbf':
            self.kernel_approx = kernel_approximation.RBFSampler(gamma=gamma, n_components=m)
        elif kernel == 'polynomial':
            self.kernel_approx = kernel_approximation.Nystroem(kernel='polynomial', gamma=-1, coef0=2, degree=-1, n_components=m)
        else:
            raise NotImplementedError

    def fit(self, X: np.ndarray):
        """
        X has shape (n, P, d1)
        """
        self.n, self.P, self.d1 = X.shape
        print('Fitting kernel with {} samples of shape {}x{}'.format(self.n, self.P, self.d1))
        self.Q = self.kernel_approx.fit_transform(np.reshape(X, (self.n*self.P, self.d1)))
        assert self.Q.shape == (self.n*self.P, self.m)

    def buid_kernel_patch_dataset(self, labels: torch.Tensor) -> data.Dataset:
        assert labels.size() == (self.n,)
        Z = np.reshape(self.Q, (self.n, self.P, self.m))
        Z_tensor = torch.from_numpy(Z).float()
        dataset = data.TensorDataset(Z_tensor, labels)
        print("Built kernel matrix as dataset with {} samples of shape {}x{}".format(len(dataset), self.P, self.m))
        return dataset
    
    def transform(self, X: np.ndarray) -> torch.Tensor:
        """
        For prediction
        X has shape (b, P, d1)
        """
        assert X.shape[1] == self.P and X.shape[2] == self.d1
        b = X.shape[0]
        transformed = self.kernel_approx.transform(np.reshape(X, (b*self.P, self.d1)))
        return torch.from_numpy(np.reshape(transformed, (b, self.P, self.m))).float()

        
        

