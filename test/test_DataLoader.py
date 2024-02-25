import unittest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.impute import SimpleImputer
from typing import Sequence, Union, Tuple
from source.hyper import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader()

    def test_load_data_file_not_found(self):
        file_path = "non_existent_file.csv"
        self.assertIsNone(self.data_loader.load_data(file_path, []))

    def test_load_data_invalid_file(self):
        file_path = "invalid_file.txt"
        self.assertIsNone(self.data_loader.load_data(file_path, []))

    def test_preprocess_data(self):
        data = pd.DataFrame({'feature1': ['Y', 'N', 'Y'], 'feature2': [1, 2, 3], 'target': [1, 0, 1]})
        X_expected = pd.DataFrame({'feature2': [1, 2, 3], 'feature1_N': [0, 1, 0], 'feature1_Y': [1, 0, 1]})
        y_expected = pd.Series([1, 0, 1])

        X_actual, y_actual = self.data_loader.preprocess_data(data, 'target')

        assert_frame_equal(X_expected, X_actual)
        assert_series_equal(y_expected, y_actual)

if __name__ == '__main__':
    unittest.main()
