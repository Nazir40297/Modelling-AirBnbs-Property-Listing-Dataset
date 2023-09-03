import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import precision_score, r2_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from torch.utils.tensorboard import SummaryWriter

from tabular_data import load_airbnb

writer = SummaryWriter()

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, hyperparameters):
    """
    Custom hyperparameter tuning for a regression model.

    This function takes a regression model class, training and validation data, and a list of hyperparameters to search through.
    It trains multiple models with different hyperparameters and selects the best model based on the validation performance.

    Parameters:
    - model_class (class): The class of the regression model to be tuned.
    - X_train (array-like): The feature matrix of the training data.
    - y_train (array-like): The target vector of the training data.
    - X_val (array-like): The feature matrix of the validation data.
    - y_val (array-like): The target vector of the validation data.
    - hyperparameters (list of dict): A list of dictionaries, where each dictionary represents a set of hyperparameters to try.

    Returns:
    - best_model (object): The best-trained regression model based on validation performance.
    - best_hyperparameters (dict): The set of hyperparameters that resulted in the best model.
    - performance_metrics (dict): A dictionary containing the best RMSE (Root Mean Square Error) and R^2 (R-squared) values.
    """
    best_model = None
    best_hyperparameters = None
    performance_metrics = {
        "Best RMSE": float("inf"), # Initialize with positive infinity, so any RMSE is better.
        "BEst R^2": -float("inf") # Initialize with negative infinity, so any R^2 is better.
    }

    # Iterate through the provided hyperparameters
    for params in hyperparameters:
        model = model_class(**params) # Create a model instance with the given hyperparameters
        # Train the model on the training data
        model.fit(X_train, y_train)

        # Calculate RMSE and R^2 on validation set
        y_val_pred = model.predict(X_val)
        validation_rmse = MSE(y_val, y_val_pred, squared = False) # Calculate RMSE
        r_squared = r2_score(y_val, y_val_pred) # Calculate R^2

        # Check if model has better RMSE 
        if validation_rmse < performance_metrics['Best RMSE']:
            best_model = model
            best_hyperparameters = params
            performance_metrics = {"Best RMSE": validation_rmse, "Best R^2": r_squared}

    return best_model, best_hyperparameters, performance_metrics


def tune_regression_model_hyperparameters(model, X_train, y_train, X_val, y_val, param_grid, scoring = 'neg_root_mean_squared_error'):
    """
    Tune hyperparameters for a regression model using GridSearchCV.

    This function uses GridSearchCV to perform hyperparameter tuning for a regression model. It takes the base model,
    training data, validation data, a parameter grid, and a scoring metric. It uses cross-validation to search through
    the parameter grid and returns the best model, its hyperparameters, and performance metrics.

    Parameters:
    - model: The base regression model to be tuned.
    - X_train: The feature matrix of the training data.
    - y_train: The target vector of the training data.
    - X_val: The feature matrix of the validation data.
    - y_val: The target vector of the validation data.
    - param_grid: A dictionary specifying the hyperparameters and their possible values to search.
    - scoring: The scoring metric used to evaluate model performance during hyperparameter tuning. Default is 'neg_root_mean_squared_error'.

    Returns:
    - best_model: The best-trained regression model based on validation performance.
    - best_hyperparameters: The set of hyperparameters that resulted in the best model.
    - perf_metrics: A dictionary containing the best RMSE (Root Mean Square Error) and R^2 (R-squared) values.
    """
    # Create a GridSearchCV object with the provided model, parameter grid, and scoring metric
    grid_search = GridSearchCV(model, param_grid, scoring = scoring, cv = 5)
    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Make predictions on the validation set using the best model
    y_val_pred = best_model.predict(X_val)

    # Calculate RMSE and R^2 for the validation predictions
    perf_metrics = {
        'Best RMSE' : MSE(y_val, y_val_pred, squared = False),
        'Best R^2' : r2_score(y_val, y_val_pred)
    }    
    return best_model, best_hyperparameters, perf_metrics


def save_model(results, model, folder):
    """
    Save a machine learning model, its hyperparameters, and performance metrics to specified files.

    This function takes the results of a machine learning model evaluation, the model name, and a folder path.
    It saves the model, hyperparameters, and performance metrics to separate files within the specified folder.

    Parameters:
    - results: A tuple containing model, hyperparameters, and performance metrics.
    - model: A string specifying the model name.
    - folder: The folder path where the model and related files will be saved.

    Returns:
    None
    """
    # Saving the machine learning model into a joblib file
    model_path = folder + model + "/model.joblib"
    joblib.dump(results[0], model_path)

    # Save hyperparameters to a JSON file
    hyperparameters_path = folder + model + "/hyperparameters.json"
    with open(hyperparameters_path, 'w') as f:
        json.dump(results[1], f)

    # Save performance metrics to a JSON file
    metrics_path = folder + model + "/metrics.json"
    with open(metrics_path , 'w') as f:
        json.dump(results[2], f)


def evaluate_all_models():
    """
    Evaluate various regression models using hyperparameter tuning.

    This function evaluates multiple regression models, including SGDRegressor, DecisionTreeRegressor,
    RandomForestRegressor, and GradientBoostingRegressor. It performs hyperparameter tuning for each model,
    saves the best models with their details, and stores them in specified directories.

    Parameters:
    None

    Returns:
    None
    """
    # Initialise an SGDRegressor model
    model = SGDRegressor()

    # Define hyperparameters for SGDRegressor
    hyperparameters = [
    {"alpha": 0.01, "l1_ratio": 0.5, "penalty": "l1"},
    {"alpha":0.001, "l1_ratio": 0.3, "penalty": "l2"},
    {"alpha":0.001, "l1_ratio":0.6, "penalty": "elasticnet"},
    {"alpha": 0.1, "l1_ratio": 0.7, "penalty": "l2"},
    {"alpha":0.1, "l1_ratio": 0.6, "penalty": "l1"},
    {"alpha": 0.001, "l1_ratio":0.7}
]
    # Perform hyperparameter tuning for SGDRegressor using custom built function
    best = custom_tune_regression_model_hyperparameters(SGDRegressor, X_train, y_train, X_val, y_val,hyperparameters)

    param_grid = {
    'alpha' : [0.001, 0.05, 0.1, 0.2, 0.25, 0.35, 0.45],
    'l1_ratio' : [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8],
    'penalty' : ['l1', 'l2', 'elasticnet', None]
    }

    # Perform hyperparameter tuning for SGDRegressor using GridSearchCV
    best2 = tune_regression_model_hyperparameters(model, X_train, y_train, X_val, y_val, param_grid)

    # Save the best SGDRegressor model
    save_model(best2, 'linear_regression', 'models/regression/')

    # Initialise a DecisionTreeRegressor model
    decision_tree = DecisionTreeRegressor()

    # Define hyperparameters for DecisionTreeRegressor
    hyperparameters = {
        'max_depth' : [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split' : [2, 3, 4, 6, 5, 7, 8, 9, 10],
        'min_samples_leaf' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

    # Perform hyperparameter tuning for DecisionTreeRegressor using GridSearchCV
    best3 = tune_regression_model_hyperparameters(decision_tree, X_train, y_train, X_val, y_val, hyperparameters, scoring = 'neg_root_mean_squared_error')

    # Save the best DecisionTreeRegressor model
    save_model(best3, 'decision_tree', 'models/regression/')

    # Define hyperparameters for RandomForestRegressor
    hyperparameters2 = {
        'n_estimators' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
        'max_depth' : [None, 5, 10],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
        'max_features' : [None, 'sqrt']
    }

    # Initialise RandomForestRegressor model
    random_forest = RandomForestRegressor()

    # Perform hyperparameter tuning for RandomForestRegressor using GridSearchCV
    best4 = tune_regression_model_hyperparameters(random_forest, X_train, y_train, X_val, y_val, hyperparameters2, scoring = 'neg_root_mean_squared_error')

    # Save the best RandomForestRegressor model
    save_model(best4, 'random_forest_regressor', 'models/regression/')

    # Define the hyperparameters for GradientBoostingRegressor
    param_grid2 = {
        'n_estimators': [20, 50, 60],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 5, 7],
        'min_samples_split': [5, 8, 10],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2']
    }

    # Initialise the GradientBoostingRegressor model
    gradient_boosting = GradientBoostingRegressor()

    # Perform hyperparameter tuning for GradientBoostingRegressor using GridSearchCV
    best5 = tune_regression_model_hyperparameters(gradient_boosting, X_train, y_train, X_val, y_val, param_grid2, scoring = 'neg_root_mean_squared_error')

    # Save the best GradientBoostingRegressor model
    save_model(best5, 'gradient_boosting_regressor', 'models/regression/')


def find_best_model(main, task_folder):
    """
    Search for the best model within a directory structure.

    This function iterates through a directory structure containing saved models, hyperparameters, and metrics.
    It loads each model, hyperparameters, and metrics and compares their performance to determine the best one.

    Parameters:
    main (str): The main directory containing task folders.
    task_folder (str): The specific task folder (e.g., 'regression' or 'classification').

    Returns:
    tuple: A tuple containing the best model, best hyperparameters, and best metrics.
    """
    best_model = None
    best_hyperparameters = None
    best_metrics = None

    # Create the path to the task folder
    tsk_folder = os.path.join(main, task_folder)

    # Get subfolders within the task folder
    subfolder = [folder for folder in os.listdir(tsk_folder)
                 if os.path.isdir(os.path.join(tsk_folder,folder))]

    for subfolders in subfolder:

        # Create the path to the subfolder
        subfolders_path = os.path.join(tsk_folder, subfolders)

        # Load the saved model
        model_path = os.path.join(subfolders_path, 'model.joblib')
        model = joblib.load(model_path)

        # Load hyperparamters from the JSON file
        hyperparameters_path = os.path.join(subfolders_path, 'hyperparameters.json')
        with open(hyperparameters_path, 'r') as f:
            hyperparameters = json.load(f)
        
        # Load metrics from the JSON file
        metrics_path = os.path.join(subfolders_path, 'metrics.json')
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Compare models and metrics based on the task
        if task_folder == 'regression':
            if best_model is None or metrics['Best RMSE'] < best_metrics['Best RMSE']:
                best_model = model
                best_hyperparameters = hyperparameters
                best_metrics = metrics
        if task_folder == 'classification':
            if best_model is None or metrics['Validation_Accuracy'] > best_metrics['Validation_Accuracy']:
                best_model = model
                best_hyperparameters = hyperparameters
                best_metrics = metrics

    return best_model, best_hyperparameters, best_metrics

def split_data(X, y):
    '''
    Split data into train, validation, and test sets with an 80-10-10 split ratio.

    This function takes features (X) and labels (y) and performs a custom train-validation-test split
    with a split ratio of 80% training, 10% validation, and 10% test data.

    Parameters:
        X (array-like): The feature dataset.
        y (array-like): The corresponding labels.

    Returns:
        X_train (array-like): The training feature dataset.
        X_val (array-like): The validation feature dataset.
        X_test (array-like): The test feature dataset.
        y_train (array-like): The training labels.
        y_val (array-like): The validation labels.
        y_test (array-like): The test labels.
    '''
     # Split data into 80% train and 20% (10% validation + 10% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42)

    # Split the remaining 20% into 10% validation and 10% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    # Loads the data from the load_airbnb function and splits it into features and labels
    X, y = load_airbnb('Price_Night')

    # Split the data into training, validation and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)    

    # Evaluates all models and saves the model, hyperparameters and metrics in their respective folders
    evaluate_all_models()

    # Extracts, evaluates and returns the best model
    best_model = find_best_model('models', 'regression')
    
    # Prints best model
    print(best_model)
    