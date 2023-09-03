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
    '''This is a custom PyTorch Dataset class. 
    It takes features and labels as inputs, converts them to PyTorch tensors, and provides methods for data loading.'''
    def __init__(self, features, labels):
        super().__init__()
        self.features = torch.tensor(features.values, dtype = torch.float32)
        self.labels = torch.tensor(labels.values, dtype = torch.float32).reshape(-1,1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NN(nn.Module):
    '''This is a custom neural network class defined using PyTorch's nn.Module. 
    It's used for creating neural network models with a specified number of hidden layers and units.'''
    def __init__(self, input_size, config):
        super(NN, self).__init__()
        self.input_size = input_size
        self.config = config

        layers = []
        prev_size = input_size
        for size in self.config['hidden_layer_width']:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def train(model, train_dataloader, validation_dataloader, num_epochs, config, device):
    '''This function handles the training of a PyTorch model. 
    It takes the model, training and validation dataloaders, number of epochs, and training configurations. 
    It calculates the loss, backpropagates, and updates the weights during training.'''
    criterion = nn.MSELoss()
    optimiser = torch.optim.__dict__[config['optimiser']](model.parameters(), lr = config['learning_rate'])

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimiser.zero_grad()

            output = model(batch_features)
            loss = criterion(output, batch_labels)

            loss.backward()
            optimiser.step()

            writer.add_scalar('Loss/train', loss.item(), epoch)

        model.eval()
        with torch.no_grad():
            for val_features, val_labels in validation_dataloader:
                val_output = model(val_features)
                val_loss = criterion(val_output, val_labels)
                validation_loss = val_loss.item()
            
                writer.add_scalar('Loss/validation', validation_loss, epoch)
    
    end_time = time.time()
    training_duration = end_time - start_time

    model.eval()
    start_time2 = time.time()

    for batch_features, batch_labels in validation_dataloader:
        with torch.no_grad():
            output = model(batch_features)
    
    end_time2 = time.time()
    inference_duration = end_time2 - start_time2
    inference_latency = inference_duration / (len(validation_dataloader) * len(batch_features))

    writer.close()

    return model, training_duration, inference_latency

def get_nn_config(config_path):
    '''This function loads a neural network configuration from a YAML file. 
    It's used to get the configuration parameters for the neural network model.'''
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def save_pytorch_model(model, hyperparameters, metrics, folder):
    '''This function saves a PyTorch model, its hyperparameters, and metrics to a specified folder. 
    If the model is an instance of nn.Module, its state dictionary is saved.'''
    if isinstance(model, torch.nn.Module):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        folder = os.path.join(folder, current_time)
        os.makedirs(folder, exist_ok = True)

        model_path = os.path.join(folder, 'model.pt')
        torch.save(model.state_dict(), model_path)

    hyperparameters_path = os.path.join(folder, 'hyperparameters.json')
    with open(hyperparameters_path, 'w') as f:
        json.dump(hyperparameters, f)
    
    metrics_path = os.path.join(folder, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

def evaluate_pytorch_model(model, dataloader, device):
    '''This function evaluates a PyTorch model using a specified dataloader. 
    It calculates RMSE and R^2 for regression tasks.'''
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            predictions = model(batch_features)
            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = MSE(y_true, y_pred, squared = False)
    r2 = r2_score(y_true, y_pred)

    return rmse, r2

def generate_nn_configs():
    '''This function generates a list of different neural network configurations. 
    It iterates over different combinations of hidden layer widths, depths, and learning rates.'''
    configs = []

    hidden_layer_widths = [32, 64, 128]
    depths = [1, 2, 3, 4]
    learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    for width in hidden_layer_widths:
        for depth in depths:
            for lr in learning_rate:
                config = {
                    'optimiser' : 'Adam',
                    'learning_rate' : lr,
                    'hidden_layer_width' : [width] * depth
                }
                configs.append(config)
    
    return configs

def find_best_nn(train_dataloader, validation_dataloader, num_epochs, folder):
    '''This function performs an exhaustive search for the best neural network configuration. 
    It trains models with different configurations, evaluates them, and selects the one with the best performance.'''
    configs = generate_nn_configs()
    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_val_rmse = float('inf')

    for idx, config in enumerate(configs):
        print(f"Training model {idx + 1}/{len(configs)}...")

        input_size = 11
        model = NN(input_size, config)

        trained_model, training_duration, inference_latency = train(model, train_dataloader, validation_dataloader, num_epochs = num_epochs, config = config, device = device)

        train_rmse, train_r2 = evaluate_pytorch_model(trained_model, train_dataloader, device)
        val_rmse, val_r2 = evaluate_pytorch_model(trained_model, validation_dataloader, device)
        test_rmse, test_r2 = evaluate_pytorch_model(trained_model, test_dataloader, device)

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
    