import unittest
import torch
import torch.nn as nn
import layers
import kernels
import numpy as np

class BasicTestCase(unittest.TestCase):
    def testCCNNLayerLinearDim(self):
        l = layers.CCNNLayerLinear(5, 5, 2, 1)
        z = torch.zeros((5, 5, 5))
        output = l.forward(z)
        self.assertEqual(output.size(), (5, 2))

    def testApproxKernelMachineDim(self):
        n, P, d1, m, b = 100, 10, 10, 10, 5
        kernel = kernels.ApproxKernelMachine('rbf', m)
        X = np.random.randn(n, P, d1)
        batch = np.random.randn(b, P, d1)
        kernel.fit(X)
        dset = kernel.buid_kernel_patch_dataset(torch.Tensor(np.random.randn(n)))
        self.assertEqual(dset[0][0].size(), (P, m))
        output = kernel.transform(batch)
        self.assertEqual(output.size(), (b, P, m))

if __name__ == '__main__':
    unittest.main()
