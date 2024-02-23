import unittest
import numpy as np
from source.hyper_v4_OOP import GeneticAlgorithm 

class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.genetic_algo = GeneticAlgorithm()

    def test_initialize_population(self):
        population_size = 10
        num_hyperparameters = 5

        population = self.genetic_algo.initialize_population(population_size, num_hyperparameters)

        # Check if population is a list
        self.assertIsInstance(population, list)

        # Check if each individual in the population is a numpy array
        for individual in population:
            self.assertIsInstance(individual, np.ndarray)

        # Check if the population size matches the specified size
        self.assertEqual(len(population), population_size)

        # Check if the hyperparameters have the correct shape
        for individual in population:
            self.assertEqual(individual.shape, (num_hyperparameters,))

    def test_select_parents(self):
        population = [np.random.rand(5) for _ in range(10)]
        fitness_values = np.random.rand(10)

        parents = self.genetic_algo.select_parents(population, fitness_values)

        # Check if parents is a list with two numpy arrays
        self.assertIsInstance(parents, list)
        self.assertEqual(len(parents), 2)
        for parent in parents:
            self.assertIsInstance(parent, np.ndarray)
            self.assertEqual(parent.shape, (5,))

    def test_crossover(self):
        parents = [np.random.rand(5), np.random.rand(5)]

        offspring = self.genetic_algo.crossover(parents)

        # Check if offspring is a list with two numpy arrays
        self.assertIsInstance(offspring, list)
        self.assertEqual(len(offspring), 2)
        for child in offspring:
            self.assertIsInstance(child, np.ndarray)
            self.assertEqual(child.shape, (5,))

    def test_mutate(self):
        individual = np.random.rand(5)

        mutated_individual = self.genetic_algo.mutate(individual)

        # Check if mutated_individual is a numpy array
        self.assertIsInstance(mutated_individual, np.ndarray)
        self.assertEqual(mutated_individual.shape, (5,))

if __name__ == '__main__':
    unittest.main()
