import unittest
import torch
import torch.nn as nn
from source.hyper import NeuralNetwork  

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_sizes = [20, 30]
        self.output_size = 1
        self.network = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)

    def test_forward(self):
        input_tensor = torch.randn(32, self.input_size)  # Example input tensor with batch size 32
        output_tensor = self.network(input_tensor)

        # Assert the output tensor has the correct shape
        self.assertEqual(output_tensor.shape, (32, self.output_size))

        # Assert the output tensor is in the correct range (between 0 and 1)
        self.assertTrue(torch.all(output_tensor >= 0))
        self.assertTrue(torch.all(output_tensor <= 1))

if __name__ == '__main__':
    unittest.main()
