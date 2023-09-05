import datetime
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from modelling_classification import combine_classes
from modelling_regression import split_data
from tabular_data import load_airbnb

writer = SummaryWriter()

class AirbnbNightlyPriceRegressionDataset(Dataset):
    '''
    A custom PyTorch Dataset class for Airbnb nightly price regression.

    This class takes features and labels as inputs, converts them to PyTorch tensors, and provides methods for data loading.

    Parameters:
        features (DataFrame): Input features in a pandas DataFrame.
        labels (DataFrame): Labels corresponding to the features in a pandas DataFrame.

    Attributes:
        features (Tensor): Tensor containing input features of float32 data type.
        labels (Tensor): Tensor containing labels corresponding to the features, reshaped as (n_samples, 1).

    Methods:
        __init__(self, features, labels): Initialises the dataset with features and labels.
        __len__(self): Returns the total number of samples in the dataset.
        __getitem__(self, idx): Retrieves a specific sample from the dataset by index.
    '''
    def __init__(self, features, labels):
        '''
        Initialises the dataset with features and labels.

        Parameters:
            features (DataFrame): Input features in a pandas DataFrame.
            labels (DataFrame): Labels corresponding to the features in a pandas DataFrame.
        '''
        super().__init__()

        # Convert features and labels to PyTorch sensors of float32 data type
        self.features = torch.tensor(features.values, dtype = torch.float32)
        self.labels = torch.tensor(labels.values, dtype = torch.float32).reshape(-1,1)
    
    def __len__(self):
        '''
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        '''
        return len(self.features)
    
    def __getitem__(self, idx):
        '''
        Retrieves a specific sample from the dataset by index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the feature tensor and label tensor of the sample.
        '''
        return self.features[idx], self.labels[idx]
    

class NN(nn.Module):
    '''
    A custom neural network class defined using PyTorch's nn.Module.

    Parameters:
        input_size (int): The number of input features.
        config (dict): A configuration dictionary specifying the architecture of the neural network.

    Attributes:
        input_size (int): The number of input features.
        config (dict): A configuration dictionary specifying the architecture of the neural network.
        layers (nn.Sequential): A sequential container for the neural network layers.

    Methods:
        __init__(self, input_size, config): Initialises the neural network.
        forward(self, x): Defines the forward pass of the neural network.
    '''
    def __init__(self, input_size, config):
        '''
        Initialises the neural network.

        Parameters:
            input_size (int): The number of input features.
            config (dict): A configuration dictionary specifying the architecture of the neural network.
        '''
        super(NN, self).__init__()
        self.input_size = input_size
        self.config = config

        layers = []
        prev_size = input_size

        # Create the hidden layers based on the configuration
        for size in self.config['hidden_layer_width']:
            layers.append(nn.Linear(prev_size, size)) # Add a linear layer 
            layers.append(nn.ReLU()) # Add a ReLU activation function 
            prev_size = size

        layers.append(nn.Linear(prev_size, 1)) # Add the output layer
        self.layers = nn.Sequential(*layers) # Create a sequential neural netowrk model

    def forward(self, x):
        '''
        Defines the forward pass of the neural network.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor produced by the neural network.
        '''
        return self.layers(x)

def train(model, train_dataloader, validation_dataloader, num_epochs, config, device):
    '''
    Train a PyTorch model.

    This function handles the training of a PyTorch model. It takes the model, training and validation dataloaders,
    number of epochs, and training configurations. It calculates the loss, backpropagates, and updates the weights during training.

    Parameters:
        model (nn.Module): The PyTorch model to be trained.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        validation_dataloader (DataLoader): DataLoader for the validation dataset.
        num_epochs (int): Number of training epochs.
        config (dict): Configuration settings for training including 'optimiser' and 'learning_rate'.
        device (str): The device to run training on ('cpu' or 'cuda').

    Returns:
        nn.Module: The trained PyTorch model.
        float: Training duration in seconds.
        float: Inference latency (average time per inference) in seconds.
    '''
    # Define loss criterion and optimiser
    criterion = nn.MSELoss()
    optimiser = torch.optim.__dict__[config['optimiser']](model.parameters(), lr = config['learning_rate'])

    # Measure the start time for training
    start_time = time.time()

    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        for batch_features, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimiser.zero_grad()

            # Forward pass
            output = model(batch_features)
            # Calculate loss
            loss = criterion(output, batch_labels)

            # Backpropogation and weight update
            loss.backward()
            optimiser.step()

            # Log the training loss
            writer.add_scalar('Loss/train', loss.item(), epoch)

        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            for val_features, val_labels in validation_dataloader:
                val_output = model(val_features)
                val_loss = criterion(val_output, val_labels)
                validation_loss = val_loss.item()

                # Log the validation loss
                writer.add_scalar('Loss/validation', validation_loss, epoch)
    
    # Measure the end time for training
    end_time = time.time()
    training_duration = end_time - start_time

    # Set the model to evaluation mode again for inference measurement
    model.eval()
    start_time2 = time.time()

    for batch_features, batch_labels in validation_dataloader:
        with torch.no_grad():
            output = model(batch_features)
    
    # Measure the end time for inference
    end_time2 = time.time()
    inference_duration = end_time2 - start_time2
    
    # Calculate inference latency 
    inference_latency = inference_duration / (len(validation_dataloader) * len(batch_features))

    # Close the writer for logging 
    writer.close()

    return model, training_duration, inference_latency

def get_nn_config(config_path):
    '''
    Load a neural network configuration from a YAML file.

    This function reads a YAML configuration file and returns the configuration parameters
    needed for configuring a neural network model.

    Parameters:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters for the neural network model.
    '''
    # Open and read the YAML configuration file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Return the loaded configuration
    return config

def save_pytorch_model(model, hyperparameters, metrics, folder):
    '''
    Save a PyTorch model, its hyperparameters, and metrics to a specified folder.

    This function is used to save a PyTorch model, its associated hyperparameters, and evaluation metrics
    to a designated folder. If the model is an instance of nn.Module, its state dictionary is saved.

    Parameters:
        model (nn.Module or None): The PyTorch model to be saved. Set to None if not applicable.
        hyperparameters (dict): A dictionary containing hyperparameters used for model training.
        metrics (dict): A dictionary containing evaluation metrics for the model.
        folder (str): The folder path where the model, hyperparameters, and metrics will be saved.
    '''
    # Check if the model is an instance of nn.Module
    if isinstance(model, torch.nn.Module):
        # Generate a timestamp for creating a unique folder
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        folder = os.path.join(folder, current_time)
        os.makedirs(folder, exist_ok = True)

        # Save the model's state dictionary
        model_path = os.path.join(folder, 'model.pt')
        torch.save(model.state_dict(), model_path)

    # Save hyperparameters as a JSON file
    hyperparameters_path = os.path.join(folder, 'hyperparameters.json')
    with open(hyperparameters_path, 'w') as f:
        json.dump(hyperparameters, f)
    
    # Save metrics as a JSON file
    metrics_path = os.path.join(folder, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

def evaluate_pytorch_model(model, dataloader, device):
    '''
    Evaluate a PyTorch model using a specified dataloader.

    This function evaluates a PyTorch model using a provided dataloader. It calculates the Root Mean Squared Error (RMSE)
    and R-squared (R^2) for regression tasks.

    Parameters:
        model (nn.Module): The PyTorch model to be evaluated.
        dataloader (DataLoader): The dataloader containing the evaluation data.
        device (str): The device (e.g., 'cuda' or 'cpu') on which the evaluation will be performed.

    Returns:
        float: The Root Mean Squared Error (RMSE) for the model's predictions.
        float: The R-squared (R^2) score for the model's predictions.
    '''
    # Set the model to evaluation mode
    model.eval()

    # Initialise lists to store true and predicted values
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            # Move batch data to the specified device
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Make predictions using the model
            predictions = model(batch_features)
            # Extend the lists with batch results
            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    # Convert the lists to NumPy arrays for calculations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate RMSE and R^2 scores
    rmse = MSE(y_true, y_pred, squared = False)
    r2 = r2_score(y_true, y_pred)

    return rmse, r2

def generate_nn_configs():
    '''
    Generate a list of different neural network configurations.

    This function generates a list of different neural network configurations by iterating over different combinations of 
    hidden layer widths, depths, and learning rates. It provides a list of dictionaries, each representing a unique neural
    network configuration.

    Returns:
        list: A list of dictionaries, each containing a unique neural network configuration.
    '''
    # Initialise an empty list to store the configurations
    configs = []

    # Define possible values for hidden layer widths, depths, and learning rates
    hidden_layer_widths = [32, 64, 128]
    depths = [1, 2, 3, 4]
    learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    # Generate configurations by combining different values
    for width in hidden_layer_widths:
        for depth in depths:
            for lr in learning_rate:
                # Create a configuration dictionary
                config = {
                    'optimiser' : 'Adam', # You can customise the optimiser as needed
                    'learning_rate' : lr,
                    'hidden_layer_width' : [width] * depth # Create a list of the same width for each hidden layer
                }
                configs.append(config) # Add the configuration to the list of configurations
    
    return configs

def find_best_nn(train_dataloader, validation_dataloader, num_epochs, folder):
    '''
    Find the best neural network configuration through an exhaustive search.

    This function performs an exhaustive search for the best neural network configuration. It trains models with different configurations,
    evaluates them, and selects the one with the best performance based on RMSE (Root Mean Square Error) on the validation set.

    Args:
        train_dataloader (DataLoader): DataLoader for the training dataset.
        validation_dataloader (DataLoader): DataLoader for the validation dataset.
        num_epochs (int): Number of training epochs.
        folder (str): Path to the folder where the best model and results will be saved.

    Returns:
        tuple: A tuple containing the best trained model, its hyperparameters, and evaluation metrics.
    '''
    # Generate a list of different neural network configurations
    configs = generate_nn_configs()

    # Initialise variables to store the best model, metrics, and hyperparameters
    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_val_rmse = float('inf')

    # Iterate through different configurations and train/evaluate models
    for idx, config in enumerate(configs):
        print(f"Training model {idx + 1}/{len(configs)}...")

         # Define input size based on the number of features
        input_size = 11
        # Create a neural network model with the current configuration
        model = NN(input_size, config)

        # Train the model and get training duration and inference latency
        trained_model, training_duration, inference_latency = train(model, train_dataloader, validation_dataloader, num_epochs = num_epochs, config = config, device = device)

        # Evaluate the model on training, validation, and test datasets
        train_rmse, train_r2 = evaluate_pytorch_model(trained_model, train_dataloader, device)
        val_rmse, val_r2 = evaluate_pytorch_model(trained_model, validation_dataloader, device)
        test_rmse, test_r2 = evaluate_pytorch_model(trained_model, test_dataloader, device)

        # Check if the current configuration has a lower validation RMSE
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = trained_model
            best_metrics = {
                'RMSE' : {
                    'train' : float(train_rmse),
                    'validation' : float(val_rmse),
                    'test' : float(test_rmse)
                },
                'R^2' : {
                    'train' : float(train_r2),
                    'validation' : float(val_r2),
                    'test' : float(test_r2)
                },
                'training_duration' : float(training_duration),
                'inference_latency' : float(inference_latency)
            }
            best_hyperparameters = config

    # Save the best model, hyperparameters, and metrics to the specified folder
    save_pytorch_model(best_model, best_hyperparameters, best_metrics, folder)

    return best_model, best_hyperparameters, best_metrics

if __name__ == '__main__':
    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Loads the data from the load_airbnb function and splits it into features and labels
    X, y = load_airbnb('Price_Night')

     # Split the data into training, validation and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Create the datasets
    train_dataset = AirbnbNightlyPriceRegressionDataset(X_train, y_train)
    validation_dataset = AirbnbNightlyPriceRegressionDataset(X_val, y_val)
    test_dataset = AirbnbNightlyPriceRegressionDataset(X_test, y_test)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = 64, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle = True)

    config = get_nn_config('nn_config.yaml')

    input_size = 11
    model = NN(input_size, config)

    num_epochs = 20
    trained_model, training_duration, inference_latency = train(model, train_dataloader, validation_dataloader, num_epochs = num_epochs, config = config, device = device)

    hyperparameters = config

    train_rmse, train_r2 = evaluate_pytorch_model(trained_model, train_dataloader, device)
    val_rmse, val_r2 = evaluate_pytorch_model(trained_model, validation_dataloader, device)
    test_rmse, test_r2 = evaluate_pytorch_model(trained_model, test_dataloader, device)

    metrics = {
        'RMSE': {
            'train' : float(train_rmse),
            'validation' : float(val_rmse),
            'test' : float(test_rmse)
        },
        'R^2' : {
            'train' : float(train_r2),
            'validation' : float(val_r2),
            'test' : float(test_r2)
        },
        'training_duration' : float(training_duration),
        'inference_latency' : float(inference_latency)
    }
    
    save_pytorch_model(trained_model, hyperparameters, metrics)

    best_model, best_metrics, best_hyperparameters = find_best_nn(train_dataloader, validation_dataloader, num_epochs)

    X, y = load_airbnb('bedrooms')

    # Split the data into training, validation and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    y_train_comb = combine_classes(y_train)
    y_val_comb = combine_classes(y_val)
    y_test_comb = combine_classes(y_test)

    train_dataset = AirbnbNightlyPriceRegressionDataset(X_train, y_train_comb)
    validation_dataset = AirbnbNightlyPriceRegressionDataset(X_val, y_val_comb)
    test_dataset = AirbnbNightlyPriceRegressionDataset(X_test, y_test_comb)

    train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    validation_dataloader = DataLoader(validation_dataset, batch_size = 64, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle = True)

    best_model, best_metrics, best_hyperparameters = find_best_nn(train_dataloader, validation_dataloader, num_epochs=20, folder = 'neural_networks/regression_beds')
    