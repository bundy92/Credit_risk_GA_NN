import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

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
# One-hot encode categorical variables first
categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Apply imputation on the encoded data
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)


# Split the dataset into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Train PD model (logistic regression)
pd_model = LogisticRegression()
pd_model.fit(X_train, y_train)

# Evaluate PD model
y_pred = pd_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, pd_model.predict_proba(X_test)[:, 1])
gini_coefficient = 2 * auc_roc - 1

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("AUC-ROC:", auc_roc)
print("GINI Coefficient:", gini_coefficient)

# Add calculated metrics to the dataset
data['PD'] = pd_model.predict_proba(X_imputed)[:, 1]
data['AUC_ROC'] = auc_roc
data['GINI'] = gini_coefficient

# Separate features and target variable again after adding metrics
X = data.drop(columns=[target_column])
y = data[target_column]

# Scale the data using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert scaled data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
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
hidden_sizes = [256, 128, 64, 32]  # Increase model capacity
output_size = 1

# Define genetic algorithm parameters
population_size = 100
num_generations = 100
mutation_rate = 0.01
learning_rates = [0.001, 0.01, 0.1]  # Experiment with different learning rates
num_epochs = 500

# Initialize population
population = initialize_population(population_size, len(hidden_sizes) + 1)  # Number of hyperparameters + 1 for learning rate

# Define the model
model = NeuralNetwork(input_size, hidden_sizes, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Initialize best_model_params with initial model parameters
best_model_params = model.state_dict()

# Training loop
best_auc_roc = -1
for generation in range(num_generations):
    fitness_values = []
    for individual_idx, individual in enumerate(population):
        learning_rate = individual[-1]
        model.load_state_dict(best_model_params)  # Restore best model parameters
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Update optimizer with new learning rate

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
            torch.save(best_model_params, 'best_model_state_dict.pth')  # Save the best model

        sys.stdout.write(f"\rGeneration {generation+1}/{num_generations}, Individual {individual_idx+1}/{population_size}, AUC-ROC: {auc_roc:.4f}, Best AUC-ROC: {best_auc_roc:.4f}")
        sys.stdout.flush()

    print(f"\nGeneration {generation+1}/{num_generations}, Best AUC-ROC: {best_auc_roc:.4f}")

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
