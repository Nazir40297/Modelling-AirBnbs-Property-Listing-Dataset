import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import joblib
import json
import os
import yaml
import datetime
import time
from tabular_data import load_airbnb


writer = SummaryWriter()

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, hyperparameters):
    '''This function performs a custom hyperparameter tuning for a regression model. It takes as input the model class, 
    training and validation data, and a list of hyperparameters. It trains the model with each set of hyperparameters, 
    calculates RMSE and R^2 on the validation set, and returns the best model, its hyperparameters, 
    and the best performance metrics.'''
    best_model = None
    best_hyperparameters = None
    performance_metrics = {
        "Best RMSE": float("inf"),
        "BEst R^2": -float("inf")
    }

    for params in hyperparameters:
        model = model_class(**params)
        # Train the model
        model.fit(X_train, y_train)

        # Calculate RMSE and R^2 on validation set
        y_val_pred = model.predict(X_val)
        validation_rmse = MSE(y_val, y_val_pred, squared = False)
        r_squared = r2_score(y_val, y_val_pred)

        # Check if model has better RMSE 
        if validation_rmse < performance_metrics['Best RMSE']:
            best_model = model
            best_hyperparameters = params
            performance_metrics = {"Best RMSE": validation_rmse, "Best R^2": r_squared}

    return best_model, best_hyperparameters, performance_metrics


def tune_regression_model_hyperparameters(model, X_train, y_train, X_val, y_val, param_grid, scoring = 'neg_root_mean_squared_error'):
    '''This function uses GridSearchCV to perform hyperparameter tuning for a regression model. 
    It takes the base model, training data, validation data, a parameter grid, and a scoring metric. 
    It uses cross-validation to search through the parameter grid, returns the best model, its hyperparameters, and performance metrics.'''
    grid_search = GridSearchCV(model, param_grid, scoring = scoring, cv = 5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    y_val_pred = best_model.predict(X_val)
    perf_metrics = {
        'Best RMSE' : MSE(y_val, y_val_pred, squared = False),
        'Best R^2' : r2_score(y_val, y_val_pred)
    }    
    return best_model, best_hyperparameters, perf_metrics


def save_model(results, model, folder):
    '''This function saves the trained model, its hyperparameters, and performance metrics to files. 
    It takes the results from hyperparameter tuning, the model name, and a folder path. 
    It creates a subfolder with the model name and saves the model, hyperparameters, and metrics as JSON files.'''
    # Saving the model into a joblib file
    model_path = folder + model + "/model.joblib"
    joblib.dump(results[0], model_path)

    # Save hyperparameters
    hyperparameters_path = folder + model + "/hyperparameters.json"
    with open(hyperparameters_path, 'w') as f:
        json.dump(results[1], f)

    # Save performance metrics
    metrics_path = folder + model + "/metrics.json"
    with open(metrics_path , 'w') as f:
        json.dump(results[2], f)


def evaluate_all_models():
    ''' This function evaluates various regression models using hyperparameter tuning. 
    It evaluates models like SGDRegressor, DecisionTreeRegressor, RandomForestRegressor, and GradientBoostingRegressor. 
    It calls the hyperparameter tuning functions for each model and saves the best models with their details.'''
    model = SGDRegressor()

    hyperparameters = [
    {"alpha": 0.01, "l1_ratio": 0.5, "penalty": "l1"},
    {"alpha":0.001, "l1_ratio": 0.3, "penalty": "l2"},
    {"alpha":0.001, "l1_ratio":0.6, "penalty": "elasticnet"},
    {"alpha": 0.1, "l1_ratio": 0.7, "penalty": "l2"},
    {"alpha":0.1, "l1_ratio": 0.6, "penalty": "l1"},
    {"alpha": 0.001, "l1_ratio":0.7}
]

    best = custom_tune_regression_model_hyperparameters(SGDRegressor, X_train, y_train, X_val, y_val,hyperparameters)

    param_grid = {
    'alpha' : [0.001, 0.05, 0.1, 0.2, 0.25, 0.35, 0.45],
    'l1_ratio' : [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8],
    'penalty' : ['l1', 'l2', 'elasticnet', None]
    }

    best2 = tune_regression_model_hyperparameters(model, X_train, y_train, X_val, y_val, param_grid)

    save_model(best2, 'linear_regression', 'models/regression/')

    decision_tree = DecisionTreeRegressor()

    hyperparameters = {
        'max_depth' : [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split' : [2, 3, 4, 6, 5, 7, 8, 9, 10],
        'min_samples_leaf' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

    best3 = tune_regression_model_hyperparameters(decision_tree, X_train, y_train, X_val, y_val, hyperparameters, scoring = 'neg_root_mean_squared_error')

    save_model(best3, 'decision_tree', 'models/regression/')

    hyperparameters2 = {
        'n_estimators' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50],
        'max_depth' : [None, 5, 10],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
        'max_features' : [None, 'sqrt']
    }

    random_forest = RandomForestRegressor()

    best4 = tune_regression_model_hyperparameters(random_forest, X_train, y_train, X_val, y_val, hyperparameters2, scoring = 'neg_root_mean_squared_error')

    save_model(best4, 'random_forest_regressor', 'models/regression/')

    param_grid2 = {
        'n_estimators': [20, 50, 60],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [3, 5, 7],
        'min_samples_split': [5, 8, 10],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2']
    }

    gradient_boosting = GradientBoostingRegressor()

    best5 = tune_regression_model_hyperparameters(gradient_boosting, X_train, y_train, X_val, y_val, param_grid2, scoring = 'neg_root_mean_squared_error')

    save_model(best5, 'gradient_boosting_regressor', 'models/regression/')


def find_best_model(main, task_folder):
    '''This function searches for the best model within a directory structure. 
    It loads each model, hyperparameters, and metrics and compares their performance to determine the best one.'''
    best_model = None
    best_hyperparameters = None
    best_metrics = None

    tsk_folder = os.path.join(main, task_folder)

    subfolder = [folder for folder in os.listdir(tsk_folder)
                 if os.path.isdir(os.path.join(tsk_folder,folder))]

    for subfolders in subfolder:

        subfolders_path = os.path.join(tsk_folder, subfolders)

        model_path = os.path.join(subfolders_path, 'model.joblib')
        model = joblib.load(model_path)

        hyperparameters_path = os.path.join(subfolders_path, 'hyperparameters.json')
        with open(hyperparameters_path, 'r') as f:
            hyperparameters = json.load(f)
        
        metrics_path = os.path.join(subfolders_path, 'metrics.json')
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
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


if __name__ == '__main__':
    # Loads the data from the load_airbnb function and splits it into features and labels
    X, y = load_airbnb('Price_Night')

    # Splits the data into traning and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = 42)

    # Evaluates all models and saves the model, hyperparameters and metrics in their respective folders
    evaluate_all_models()

    # Extracts, evaluates and returns the best model
    best_model = find_best_model('models', 'regression')

    # Prints best model
    print(best_model)
    