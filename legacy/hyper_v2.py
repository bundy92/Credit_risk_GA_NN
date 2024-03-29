# Refactored code for the main script

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please provide the correct file path.")
        exit()
    except Exception as e:
        print("An error occurred while loading the dataset:", e)
        exit()
    return data

def preprocess_data(data):
    target_column = 'cb_person_default_on_file'
    data[target_column] = data[target_column].map({'Y': 1, 'N': 0})

    X = data.drop(columns=[target_column])
    y = data[target_column]

    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)

    return X_imputed, y

def train_pd_model(X_train, y_train, best_model_params=None):
    if best_model_params:
        # Use best_model_params for training if provided
        pd_model = LogisticRegression(**best_model_params, max_iter=1000, solver='liblinear')
    else:
        # Train with default parameters if best_model_params is not provided
        pd_model = LogisticRegression(max_iter=1000, solver='liblinear')
    pd_model.fit(X_train, y_train)
    return pd_model


def evaluate_pd_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, precision, recall, f1, auc_roc

def train_neural_network(X_train, y_train, X_val, y_val, input_size, hidden_sizes, output_size,
                         learning_rate, num_epochs):
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

    # Scale the data using Min-Max scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert scaled data to PyTorch tensors and move to GPU
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1).to(device)

    # Define the model and move it to GPU
    model = NeuralNetwork(input_size, hidden_sizes, output_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_auc_roc = -1
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
            auc_roc = roc_auc_score(y_val, outputs_val.cpu().numpy())  # Move back to CPU for metric calculation

        # Update the best model
        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_model_params = model.state_dict()

        sys.stdout.write(f"\rAUC-ROC: {auc_roc:.4f}, Best AUC-ROC: {best_auc_roc:.4f}")
        sys.stdout.flush()

    #print(f"\nBest AUC-ROC: {best_auc_roc:.4f}")

    return best_model_params, best_auc_roc

def genetic_algorithm_hyperparameter_optimization(X_train, y_train, X_val, y_val, input_size, hidden_sizes,
                                                   output_size, population_size, num_generations, mutation_rate,
                                                   learning_rates, batch_sizes, num_epochs):
    def initialize_population(population_size, num_hyperparameters):
        population = []
        for _ in range(population_size):
            individual = np.random.uniform(low=0.0001, high=1, size=num_hyperparameters)
            population.append(individual)
        return population

    def select_parents(population, fitness_values):
        probabilities = np.exp(fitness_values) / np.sum(np.exp(fitness_values))
        parents_indices = np.random.choice(len(population), size=2, p=probabilities, replace=False)
        return [population[idx] for idx in parents_indices]

    def crossover(parents, crossover_rate=0.8):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, len(parents[0]))
            child1 = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
            child2 = np.concatenate((parents[1][:crossover_point], parents[0][crossover_point:]))
            return [child1, child2]
        else:
            return parents

    def mutate(individual, mutation_rate=0.1):
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                individual[i] = np.random.uniform(low=0.0001, high=1)
        return individual

    # Initialize population
    population = initialize_population(population_size, len(hidden_sizes) + 2)  # Number of hyperparameters + 2 for learning rate and batch size

    # Initialize best_model_params with initial model parameters
    best_auc_roc = -1
    best_model_params = None

    # Training loop
    for generation in range(num_generations):
        fitness_values = []
        for individual_idx, individual in enumerate(population):
            learning_rate = individual[-2]
            batch_size = int(round(individual[-1]))  # Convert batch size to integer
            model_params, auc_roc = train_neural_network(X_train, y_train, X_val, y_val, input_size, hidden_sizes,
                                                          output_size, learning_rate, batch_size, num_epochs)
            fitness_values.append(auc_roc)

            # Update the best model
            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                best_model_params = model_params

            sys.stdout.write(f"\rGeneration {generation+1}/{num_generations}, Individual {individual_idx+1}/{population_size}, AUC-ROC: {auc_roc:.4f}, Best AUC-ROC: {best_auc_roc:.4f}")
            sys.stdout.flush()

        # Select parents and create new population
        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population, fitness_values)
            offspring = crossover(parents)
            offspring = [mutate(child, mutation_rate) for child in offspring]
            new_population.extend(offspring)

        population = new_population

    return best_model_params, best_auc_roc

# Function to plot line plot
def plot_line_plot(initial_values, optimized_values, hyperparameters):
    plt.figure(figsize=(10, 6))
    plt.plot(hyperparameters, initial_values, marker='o', color='blue', label='Initial Values')
    plt.plot(hyperparameters, optimized_values, marker='o', color='orange', label='Optimized Values')
    plt.xlabel('Hyperparameters')
    plt.ylabel('Values')
    plt.title('Hyperparameter Values Before and After Optimization')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot()

# Function to plot bar chart
def plot_bar_chart(initial_values, optimized_values, hyperparameters):
    plt.figure(figsize=(10, 6))
    x = range(len(hyperparameters))
    plt.bar(x, initial_values, width=0.4, color='blue', label='Initial Values')
    plt.bar([i+0.4 for i in x], optimized_values, width=0.4, color='orange', label='Optimized Values')
    plt.xlabel('Hyperparameters')
    plt.ylabel('Values')
    plt.title('Hyperparameter Values Before and After Optimization')
    plt.xticks([i + 0.2 for i in x], hyperparameters, rotation=45)
    plt.legend()
    plt.grid(axis='y')
    st.pyplot()

# Function to plot heatmap
def plot_heatmap(initial_values, optimized_values, hyperparameters):
    data = [initial_values, optimized_values]
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='YlOrBr', aspect='auto')
    plt.colorbar(label='Hyperparameter Values')
    plt.title('Hyperparameter Values Before and After Optimization')
    plt.xticks(range(len(hyperparameters)), hyperparameters, rotation=45)
    plt.yticks(range(2), ['Initial Values', 'Optimized Values'])
    st.pyplot()


#if __name__ == "__main__":

# # Load the dataset
# data = load_data("credit_risk_dataset.csv")

# # Preprocess the data
# X, y = preprocess_data(data)

# # Split the dataset into training, validation, and test sets
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# # Train PD model (logistic regression)
# pd_model = train_pd_model(X_train, y_train)

# # Evaluate PD model
# accuracy, precision, recall, f1, auc_roc = evaluate_pd_model(pd_model, X_test, y_test)
# gini_coefficient = 2 * auc_roc - 1

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)
# print("AUC-ROC:", auc_roc)
# print("GINI Coefficient:", gini_coefficient)

# # Add calculated metrics to the dataset
# data['PD'] = pd_model.predict_proba(X)[:, 1]
# data['AUC_ROC'] = auc_roc
# data['GINI'] = gini_coefficient

# # Hyperparameter optimization using genetic algorithm
# best_model_params, best_auc_roc = genetic_algorithm_hyperparameter_optimization(X_train, y_train, X_val, y_val,
#                                                                                 input_size=X_train.shape[1],
#                                                                                 hidden_sizes=[256, 128, 64, 32],
#                                                                                 output_size=1,
#                                                                                 population_size=100,
#                                                                                 num_generations=100,
#                                                                                 mutation_rate=0.01,
#                                                                                 learning_rates=[0.001, 0.01, 0.1],
#                                                                                 num_epochs=500)

# print(f"Best AUC-ROC on Test Set: {best_auc_roc:.4f}")
