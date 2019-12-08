import unittest
import torch
import torch.nn as nn
import layers

class BasicTestCase(unittest.TestCase):
    def testCCNNLayerLinearDim(self):
        l = layers.CCNNLayerLinear(5, 5, 2, 1)
        z = torch.zeros((5, 5, 5))
        output = l.forward(z)
        self.assertEqual(output.size(), (5, 2))

if __name__ == '__main__':
    unittest.main()
