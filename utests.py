import unittest
import torch
import torch.nn as nn
import layers
import main as main

class BasicTestCase(unittest.TestCase):
    def testCCNNLayerLinearDim(self):
        l = layers.CCNNLayerLinear(5, 3, 2, 1)
        z = torch.zeros((5, 5, 3, 3))
        output = l.forward(z)
        self.assertEqual(output.size(), (5, 2))

class TrainMNISTTestCase(unittest.TestCase):
    def testNoError(self):
        args = main.parser.parse_args([
            '--approx_m=2',
            '--epochs=2',
            ])
        main.mnist_experiment(args)

if __name__ == '__main__':
    unittest.main()
