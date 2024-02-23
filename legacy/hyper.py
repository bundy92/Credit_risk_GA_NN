import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load the dataset
try:
    data = pd.read_csv("credit_risk_dataset.csv")
except FileNotFoundError:
    print("Error: File not found. Please provide the correct file path.")
    exit()
except Exception as e:
    print("An error occurred while loading the dataset:", e)
    exit()

# Check if the target column exists
target_column = 'cb_person_default_on_file'
if target_column not in data.columns:
    print(f"Error: '{target_column}' column not found in the dataset. Please check the column names.")
    exit()

# Encode 'Y' as 1 and 'N' as 0 in the target column
data[target_column] = data[target_column].map({'Y': 1, 'N': 0})

# Separate features and target variable
X = data.drop(columns=[target_column])
y = data[target_column]

# Handle missing values in features by imputation
X = pd.get_dummies(X)  # One-hot encoding
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the dataset into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

# Define genetic algorithm functions

def initialize_population(population_size, num_hyperparameters):
    """
    Initialize the population with random values.
    """
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(low=0.0001, high=1, size=num_hyperparameters)
        population.append(individual)
    return population

def select_parents(population, fitness_values):
    """
    Select parent individuals based on their fitness values.
    """
    probabilities = np.exp(fitness_values) / np.sum(np.exp(fitness_values))
    parents_indices = np.random.choice(len(population), size=2, p=probabilities, replace=False)
    return [population[idx] for idx in parents_indices]

def crossover(parents, crossover_rate=0.8):
    """
    Perform crossover to create offspring from parents.
    """
    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parents[0]))
        child1 = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
        child2 = np.concatenate((parents[1][:crossover_point], parents[0][crossover_point:]))
        return [child1, child2]
    else:
        return parents

def mutate(individual, mutation_rate=0.1):
    """
    Apply mutation to an individual.
    """
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(low=0.0001, high=1)
    return individual

# Define recommended hyperparameters
input_size = X_train.shape[1]
hidden_sizes = [128, 64, 32]
output_size = 1

# Define genetic algorithm parameters
population_size = 50
num_generations = 50
mutation_rate = 0.1
learning_rate = 0.001
num_epochs = 100

# Initialize population
population = initialize_population(population_size, len(hidden_sizes) + 1)  # Number of hyperparameters + 1 for learning rate

# Define the model
model = NeuralNetwork(input_size, hidden_sizes, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
best_auc_roc = -1
best_model_params = None
for generation in range(num_generations):
    fitness_values = []
    for individual in population:
        learning_rate = individual[-1]

        # Define the model
        model = NeuralNetwork(input_size, hidden_sizes, output_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate the model on validation set
        model.eval()
        with torch.no_grad():
            outputs_val = model(X_val_tensor)
            auc_roc = roc_auc_score(y_val, outputs_val.cpu().numpy())

        fitness_values.append(auc_roc)

        # Update the best model
        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_model_params = model.state_dict()

    print(f"Generation {generation+1}/{num_generations}, Best AUC-ROC: {best_auc_roc:.4f}")

    # Select parents and create new population
    new_population = []
    for _ in range(population_size // 2):
        parents = select_parents(population, fitness_values)
        offspring = crossover(parents)
        offspring = [mutate(child, mutation_rate) for child in offspring]
        new_population.extend(offspring)

    population = new_population

# Save the best model
torch.save(best_model_params, 'best_model_state_dict.pth')

# Load the best model
model.load_state_dict(torch.load('best_model_state_dict.pth'))

# Evaluate the best model on test set
model.eval()
with torch.no_grad():
    outputs_test = model(X_test_tensor)
    auc_roc_test = roc_auc_score(y_test, outputs_test.cpu().numpy())

print(f"Best AUC-ROC on Test Set: {auc_roc_test:.4f}")
