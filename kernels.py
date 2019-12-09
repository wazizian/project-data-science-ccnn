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
            self.kernel_approx = kernel_approximation.Nystroem(kernel='rbf', gamma=gamma, n_components=m)
        elif kernel == 'polynomial':
            self.kernel_approx = kernel_approximation.Nystroem(kernel='polynomial', gamma=-1, coef0=2, degree=-1, n_components=m)
        else:
            raise NotImplementedError

    def fit(self, X: np.array):
        """
        X has shape (n, P, d1)
        """
        self.n, self.P, self.d1 = X.shape
        self.Q = self.kernel_approx.fit_transform(np.reshape(X, (self.n*self.P, self.d1)))
        assert self.Q.shape == (self.n*self.P, self.m)

    def buid_patch_dataset(self, labels: torch.Tensor) -> data.Dataset:
        assert labels.size() == (self.n,)
        Z = np.reshape(self.Q, (self.n, self.P, self.m))
        Z_tensor = torch.from_numpy(Z)
        dataset = data.TensorDataset(Z_tensor, labels)
        return dataset
    
    def transform(self, X: np.array) -> torch.Tensor:
        """
        X has shape (b, P, d1)
        """
        assert X.shape[1] == self.P and X.shape[2] == self.d1
        b = X.shape[0]
        transformed = self.kernel_approx.transform(np.reshape(X, (b*self.P, self.d1)))
        return torch.from_numpy(np.reshape(transformed, (b, self.P, self.m))) 

        
        

