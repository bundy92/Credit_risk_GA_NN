import unittest
from source.hyper import Visualization 

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.visualization = Visualization()

    def test_plot_line_plot(self):
        initial_values = [1, 2, 3, 4, 5]
        optimized_values = [2, 3, 4, 5, 6]
        hyperparameters = ['param1', 'param2', 'param3', 'param4', 'param5']
        self.visualization.plot_line_plot(initial_values, optimized_values, hyperparameters)

        # No direct test assertion for plotting functions

    def test_plot_bar_chart(self):
        initial_values = [1, 2, 3, 4, 5]
        optimized_values = [2, 3, 4, 5, 6]
        hyperparameters = ['param1', 'param2', 'param3', 'param4', 'param5']
        self.visualization.plot_bar_chart(initial_values, optimized_values, hyperparameters)

        # No direct test assertion for plotting functions

    def test_plot_heatmap(self):
        initial_values = [1, 2, 3, 4, 5]
        optimized_values = [2, 3, 4, 5, 6]
        hyperparameters = ['param1', 'param2', 'param3', 'param4', 'param5']
        self.visualization.plot_heatmap(initial_values, optimized_values, hyperparameters)

        # No direct test assertion for plotting functions

    def test_visualize_hyperparameter_space(self):
        learning_rates = [0.001, 0.01, 0.1]
        batch_sizes = [32, 64, 128]
        performance_metrics = [0.75, 0.8, 0.85]  # Dummy accuracy values
        metric_name = 'Accuracy'
        self.visualization.visualize_hyperparameter_space(learning_rates, batch_sizes, performance_metrics, metric_name)

        # No direct test assertion for plotting functions

if __name__ == '__main__':
    unittest.main()
