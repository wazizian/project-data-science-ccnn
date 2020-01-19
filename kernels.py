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
        # pinvQ = np.linalg.pinv(self.Q)
        # self.pinvQQ = pinvQ @ self.Q 
        # in general pinvQQ is extremely close to identity: 
        # np.linalg.norm(self.pinvQQ-np.identity(len(self.pinvQQ)))) ~ 1e-11
        # (this the case if Q is full-rank, ie rank Q = m (<< nP))
        # Hence though we should multiply by this matrix in trasnform
        # we do not do it to save time & memory
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
        transformed = self.kernel_approx.transform(np.reshape(X, (-1, self.X_shape[-1])))# @ self.pinvQQ.T
        return torch.from_numpy(np.reshape(transformed, (b,) + self.Q.shape[1:])).float()


class LightApproxKernelMachine:
    def __init__(self, kernel, d1, m, gamma=0.2):
        self.m = m
        self.d1 = d1
        if kernel != 'rbf':
            raise NotImplementedError
        self.kernel_approx = kernel_approximation.RBFSampler(gamma=gamma, n_components=m)
        # according to the source code of RBF sampler, this actually uses only d_1
        # and not the data
        self.kernel_approx.fit(np.zeros((1, d1)))

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (*, d1)
        out has shape (*, m)
        """
        X_view = X.view(-1, self.d1)
        transformed = self.kernel_approx.transform(X_view.numpy())
        assert transformed.shape[1] == self.m
        assert transformed.shape[0] == X_view.size(0)
        return torch.from_numpy(transformed).float().view(X.size()[:-1] + (self.m,))
