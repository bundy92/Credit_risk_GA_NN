# hyper_v3.py
# Importing dendencies
import sys
from typing import Union, Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path: str) -> Union[pd.DataFrame, None]:
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    data (pd.DataFrame or None): Loaded dataset if successful, None otherwise.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found. Please provide the correct file path.")
        exit()
    except Exception as e:
        print("An error occurred while loading the dataset:", e)
        exit()
    return data

def preprocess_data(data: pd.DataFrame, target_column: None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the input data for machine learning tasks.

    Parameters:
    data (DataFrame): Input data containing features and target column.

    Returns:
    X_imputed (DataFrame): Preprocessed feature matrix with missing values imputed.
    y (Series): Target column.
    """
    target_column = target_column
    data[target_column] = data[target_column].map({'Y': 1, 'N': 0})

    X = data.drop(columns=[target_column])
    y = data[target_column]

    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols)
 
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns)

    return X_imputed, y

def train_pd_model(X_train: np.ndarray, y_train: np.ndarray, best_model_params: Union[dict, None] = None) -> LogisticRegression:
    """
    Trains a logistic regression model using the provided training data.

    Parameters:
    X_train (array-like): Feature matrix for training.
    y_train (array-like): Target labels for training.
    best_model_params (dict or None): Parameters for the logistic regression model. If provided,
                                      these parameters will be used for training. If None, default
                                      parameters will be used.

    Returns:
    pd_model: Trained logistic regression model.
    """
    if best_model_params:
        # Use best_model_params for training if provided
        pd_model = LogisticRegression(**best_model_params, max_iter=1000, solver='liblinear')
    else:
        # Train with default parameters if best_model_params is not provided
        pd_model = LogisticRegression(max_iter=1000, solver='liblinear')
    pd_model.fit(X_train, y_train)
    return pd_model

def evaluate_pd_model(model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Evaluates a trained classifier model using various performance metrics.

    Parameters:
    model (ClassifierMixin): Trained classifier model.
    X_test (np.ndarray): Feature matrix for testing.
    y_test (np.ndarray): True labels for testing.

    Returns:
    accuracy (float): Accuracy of the model.
    precision (float): Precision of the model.
    recall (float): Recall of the model.
    f1 (float): F1 score of the model.
    auc_roc (float): Area under the ROC curve of the model.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, precision, recall, f1, auc_roc


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initialize the neural network.

        Parameters:
        input_size (int): Size of the input features.
        hidden_sizes (Union[int, List[int]]): Size(s) of the hidden layer(s).
        output_size (int): Size of the output.

        Returns:
        None
        """
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
        """
        Perform forward pass through the network.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor.
        """
        return torch.sigmoid(self.model(x)).squeeze()

def train_neural_network(X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor,
                         input_size: int, hidden_sizes: Tuple[int], output_size: int, learning_rate: float,
                         batch_size: int, num_epochs: int, device: torch.device) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Trains a neural network model using the provided training data and validates it on validation data.

    Parameters:
    X_train (torch.Tensor): Feature matrix for training.
    y_train (torch.Tensor): Target labels for training.
    X_val (torch.Tensor): Feature matrix for validation.
    y_val (torch.Tensor): Target labels for validation.
    input_size (int): Size of the input features.
    hidden_sizes (Tuple[int]): Sizes of hidden layers.
    output_size (int): Size of the output.
    learning_rate (float): Learning rate for optimization.
    batch_size (int): Number of samples per batch.
    num_epochs (int): Number of training epochs.
    device (torch.device): Device to be used for training ('cpu' or 'cuda').

    Returns:
    best_model_params (Dict[str, torch.Tensor]): Parameters of the best model based on validation performance.
    best_auc_roc (float): Best AUC-ROC score achieved on the validation set.
    """


    # Scale the data using Min-Max scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert scaled data to PyTorch tensors and move to appropriate device
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(-1).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(device)      # Ensure target size consistency

    # Define the model and move it to appropriate device
    model = NeuralNetwork(input_size, hidden_sizes, output_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_auc_roc = -1
    best_model_params = None
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)

        # # Debugging: Print shapes of outputs and y_train_tensor
        # print("Output shape:", outputs.shape)
        # print("y_train_tensor shape:", y_train_tensor.shape)
        

        loss = criterion(outputs, y_train_tensor.squeeze())
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

        sys.stdout.write(f"\rEpoch {epoch+1}/{num_epochs}, AUC-ROC: {auc_roc:.4f}, Best AUC-ROC: {best_auc_roc:.4f}")
        sys.stdout.flush()

    print("\nTraining complete.")
    return best_model_params, best_auc_roc

def genetic_algorithm_hyperparameter_optimization(self, 
                                                X_train: np.ndarray, 
                                                y_train: np.ndarray, 
                                                X_val: np.ndarray, 
                                                y_val: np.ndarray, 
                                                input_size: int, 
                                                hidden_sizes: Tuple[int],
                                                output_size: int, 
                                                population_size: int, 
                                                num_generations: int, 
                                                mutation_rate: float,
                                                learning_rates: Tuple[float, float, float],
                                                batch_sizes: Tuple[int, int, int], 
                                                num_epochs: int) -> Tuple[Dict[str, Union[int, List[float]]], float, List[float], List[float]]:
    """
    Perform hyperparameter optimization using a genetic algorithm.

    Parameters:
    X_train (np.ndarray): Training features.
    y_train (np.ndarray): Training labels.
    X_val (np.ndarray): Validation features.
    y_val (np.ndarray): Validation labels.
    input_size (int): Size of the input features.
    hidden_sizes (list): List of hidden layer sizes.
    output_size (int): Size of the output layer.
    population_size (int): Number of individuals in the population.
    num_generations (int): Number of generations for the genetic algorithm.
    mutation_rate (float): Probability of mutation.
    learning_rates (tuple): Range of learning rates to explore.
    batch_sizes (tuple): Range of batch sizes to explore.
    num_epochs (int): Number of epochs for neural network training.

    Returns:
    best_model_params (dict): Parameters of the best model found.
    best_fitness (float): Fitness value of the best model.
    initial_values (list): Initial hyperparameter values.
    optimized_values (list): Optimized hyperparameter values.
    """

    # Initialize population
    population = self.initialize_population(population_size, len(hidden_sizes) + 3)  # 3 additional hyperparameters for learning rate, batch size, and hidden layer sizes

    # Initialize best model and fitness
    best_fitness = -1
    best_model_params = None

    # Initialize initial and optimized values
    initial_values = []
    optimized_values = []

    # Training loop
    for generation in range(num_generations):
        fitness_values = []
        for individual_idx, individual in enumerate(population):
            # Extract hyperparameters from the individual
            learning_rate = learning_rates[0] + (learning_rates[1] - learning_rates[0]) * individual[-3]
            batch_size = int(round(batch_sizes[0] + (batch_sizes[1] - batch_sizes[0]) * individual[-2]))  # Convert batch size to integer
            hidden_layer_sizes = [int(round(size)) for size in hidden_sizes * individual[:-3]]  # Convert hidden layer sizes to integers
            
            # Train neural network with the extracted hyperparameters
            model_params, evaluation_metrics = train_neural_network(X_train, y_train, X_val, y_val,
                                                                                input_size, hidden_layer_sizes, output_size,
                                                                                learning_rate, batch_size, num_epochs)

            # Calculate fitness based on multiple evaluation metrics
            current_fitness = sum(evaluation_metrics.values())
            fitness_values.append(current_fitness)

            # Update the best model based on overall fitness
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_model_params = model_params
                torch.save(best_model_params, './models/best_model_state_dict.pth')  # Save the best model


            # Modify the print statement to display all evaluation metrics and best fitness
            sys.stdout.write(f"\rGeneration {generation+1}/{num_generations}, Individual {individual_idx+1}/{population_size}, Fitness: {current_fitness:.4f}, Best Fitness: {best_fitness:.4f}, Evaluation Metrics: {evaluation_metrics}")
            sys.stdout.flush()

        # Select parents and create new population
        new_population = []
        for _ in range(population_size // 2):
            parents = self.select_parents(population, fitness_values)
            offspring = self.crossover(parents)
            offspring = [self.mutate(child, mutation_rate) for child in offspring]
            new_population.extend(offspring)

        population = new_population

        # Store the initial and optimized values
        if generation == 0:
            initial_values = population[0][:-3]
        if generation == num_generations - 1:
            optimized_values = population[0][:-3]

    return best_model_params, best_fitness, initial_values, optimized_values


# Visualisation of the optimization process

# Function to plot line chart
def plot_line_plot(initial_values: List, optimized_values: List, hyperparameters: List) -> None:
    """
    Plots a line chart comparing initial and optimized values of hyperparameters.

    Parameters:
    initial_values (list): List of initial values for hyperparameters.
    optimized_values (list): List of optimized values for hyperparameters.
    hyperparameters (list): List of hyperparameters.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hyperparameters, initial_values, marker='o', color='blue', label='Initial Values')
    ax.plot(hyperparameters, optimized_values, marker='x', color='red', label='Optimized Values')
    ax.set_title('Hyperparameter Optimization')
    ax.set_xlabel('Hyperparameter')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    ax.set_xticklabels(hyperparameters, rotation=45)
    st.pyplot(fig)

# Function to plot bar chart
def plot_bar_chart(initial_values: List, optimized_values: List, hyperparameters: List) -> None:
    """
    Plots a bar chart comparing initial and optimized values of hyperparameters.

    Parameters:
    initial_values (list): List of initial values for hyperparameters.
    optimized_values (list): List of optimized values for hyperparameters.
    hyperparameters (list): List of hyperparameters.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(hyperparameters))
    width = 0.4
    ax.bar(x - width/2, initial_values, width, color='blue', label='Initial Values')
    ax.bar(x + width/2, optimized_values, width, color='red', label='Optimized Values')
    ax.set_xlabel('Hyperparameters')
    ax.set_ylabel('Values')
    ax.set_title('Hyperparameter Values Before and After Optimization')
    ax.set_xticks(x)
    ax.set_xticklabels(hyperparameters, rotation=45)
    ax.legend()
    ax.grid(axis='y')
    st.pyplot(fig)

# Function to plot heatmap
def plot_heatmap(initial_values: List, optimized_values: List, hyperparameters: List) -> None:
    """
    Plots a heatmap comparing initial and optimized values of hyperparameters.

    Parameters:
    initial_values (list): List of initial values for hyperparameters.
    optimized_values (list): List of optimized values for hyperparameters.
    hyperparameters (list): List of hyperparameters.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [initial_values, optimized_values]
    im = ax.imshow(data, cmap='YlOrBr', aspect='auto')
    ax.set_title('Hyperparameter Values Before and After Optimization')
    ax.set_xticks(np.arange(len(hyperparameters)))
    ax.set_xticklabels(hyperparameters, rotation=45)
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(['Initial Values', 'Optimized Values'])
    fig.colorbar(im, ax=ax, label='Hyperparameter Values')
    st.pyplot(fig)

def visualize_hyperparameter_space(learning_rates: List[float], batch_sizes: List[int], performance_metrics: List[float], metric_name: str = 'Accuracy') -> None:
    """
    Visualizes the hyperparameter space using a scatter plot.

    Parameters:
    learning_rates (List[float]): List of learning rates.
    batch_sizes (List[int]): List of batch sizes.
    performance_metrics (List[float]): List of performance metrics corresponding to hyperparameter combinations.
    metric_name (str): Name of the performance metric. Default is 'Accuracy'.

    Returns:
    None
    """
    # Ensure all lists have the same length
    min_length = min(len(learning_rates), len(batch_sizes), len(performance_metrics))
    learning_rates = learning_rates[:min_length]
    batch_sizes = batch_sizes[:min_length]
    performance_metrics = performance_metrics[:min_length]

    # Plot scatter plot of hyperparameter space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot data points
    ax.scatter(learning_rates, batch_sizes, performance_metrics, c=performance_metrics, cmap='viridis', s=100)

    # Set labels and title
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel(metric_name)
    ax.set_title('Hyperparameter Space Visualization')

    # Add color bar
    cbar = plt.colorbar(ax.scatter(learning_rates, batch_sizes, performance_metrics, c=performance_metrics, cmap='viridis'))
    cbar.set_label(metric_name)

    plt.show()

# Local test case for the script.
    
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
