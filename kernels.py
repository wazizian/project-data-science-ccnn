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
        X has shape (*, d1)
        self.Q has shape (*, m)
        """
        self.X_shape = X.shape
        print('Fitting kernel with samples of shape '.format(self.X_shape))
        self.Q = self.kernel_approx.fit_transform(np.reshape(X, (-1, self.X_shape[-1])))
        pinvQ = np.linalg.pinv(self.Q)
        self.pinvQQ = pinvQ @ self.Q 
        #in general pinvQQ is extremely close to identity: np.linalg.norm(self.pinvQQ-np.identity(len(self.pinvQQ)))) ~ 1e-11
        self.Q = np.reshape(self.Q, X.shape[:-1] + (self.m,))

    def buid_kernel_patch_dataset(self, labels: torch.Tensor) -> data.Dataset:
        assert labels.size() == (self.X_shape[0],)
        Z_tensor = torch.from_numpy(self.Q).float()
        dataset = data.TensorDataset(Z_tensor, labels)
        print("Built kernel matrix as dataset of shape ".format(self.Q.shape))
        return dataset
    
    def transform(self, X: np.ndarray) -> torch.Tensor:
        """
        For prediction
        X has shape (*, d1)
        """
        assert X.shape[1:] == self.X_shape[1:]
        b = X.shape[0]
        transformed = self.kernel_approx.transform(np.reshape(X, (-1, self.X_shape[-1]))) @ self.pinvQQ.T
        return torch.from_numpy(np.reshape(transformed, (b,) + self.Q.shape[1:])).float()
