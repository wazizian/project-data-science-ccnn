import numpy as np
import sklearn.metrics.pairwise

def KernelLayer:
    def __init__(self):
        pass

    def kernel(X, Y=None):
        raise NotImplementedError

        if kernel_type == 'poly':
        elif kernel_type == 'rbf':

    def train(self, X: np.array):
        self.X = X
        self.kernel_matrix = self.kernel(X, None)
        #TODO factor self.kernel_matrix
    
    def forward():
        #TODO

def InversePolyKernel(KernelLayer):
    def __init__(self):
        super(InversePolyKernel, self).__init__()
    def kernel(X, Y=None):
            return sklearn.metrics.pairwise.polynomial_kernel(X, Y=Y, degree=-1, gamma=-1, coef0=2)

def GaussianRBFKernel(KernelLayer):
    def __init__(self, gamma=0.2):
        super(GaussianRBFKernel, self).__init__()
        self.gamma = gamma
    def kernel(X, Y=None):
            return sklearn.metrics.pairwise.rbf_kernel(X, Y=Y, gamma=self.gamma)
