import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from source.hyper_v4_OOP import ModelTrainer 

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.model_trainer = ModelTrainer()

    def test_train_pd_model_default_params(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        
        model = self.model_trainer.train_pd_model(X_train, y_train)
        self.assertIsInstance(model, LogisticRegression)

    def test_train_pd_model_custom_params(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        custom_params = {'C': 0.5}
        
        model = self.model_trainer.train_pd_model(X_train, y_train, best_model_params=custom_params)
        self.assertIsInstance(model, LogisticRegression)

    def test_evaluate_pd_model(self):
        model = LogisticRegression()
        X_test = np.array([[1, 2], [3, 4], [5, 6]])
        y_test = np.array([0, 1, 0])

        accuracy, precision, recall, f1, auc_roc = self.model_trainer.evaluate_pd_model(model, X_test, y_test)

        # Perform assertions on the computed metrics
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(precision, float)
        self.assertIsInstance(recall, float)
        self.assertIsInstance(f1, float)
        self.assertIsInstance(auc_roc, float)

if __name__ == '__main__':
    unittest.main()
