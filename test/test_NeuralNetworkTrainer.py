import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from source.hyper import NeuralNetworkTrainer

class TestNeuralNetworkTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = NeuralNetworkTrainer()

    def test_train_neural_network(self):
        input_size = 10
        hidden_sizes = (20, 30)
        output_size = 1
        learning_rate = 0.001
        batch_size = 32
        num_epochs = 10
        device = torch.device("cpu")

        X_train = torch.randn(100, input_size)
        y_train = torch.randint(0, 2, (100,))
        X_val = torch.randn(50, input_size)
        y_val = torch.randint(0, 2, (50,))

        best_model_params, best_auc_roc = self.trainer.train_neural_network(
            X_train, y_train, X_val, y_val,
            input_size, hidden_sizes, output_size,
            learning_rate, batch_size, num_epochs, device
        )

        # Check if best_model_params is a dictionary
        self.assertIsInstance(best_model_params, dict)

        # Check if best_auc_roc is a float
        self.assertIsInstance(best_auc_roc, float)

if __name__ == '__main__':
    unittest.main()
