import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np 
import joblib
import json
import os
from tabular_data import load_airbnb

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, hyperparameters):
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

def evaluate_all_class_models():
    m1 = DecisionTreeClassifier()

    hyp = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
    }

    bst2 = tune_classification_model_hyperparameters(m1, X_train, X_val, y_train, y_val, hyp)

    save_model(bst2, 'decision_tree_classifier', 'models/classification/')

    random_forest_class = RandomForestClassifier()

    hyp2 = {
    'n_estimators': [100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt'],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
    }

    bst3 = tune_classification_model_hyperparameters(random_forest_class, X_train, X_val, y_train, y_val, hyp2)

    save_model(bst3, 'random_forest_classifier', 'models/classification/')

    gradient_boosting_class = GradientBoostingClassifier()

    hyp3 = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt'],
    'subsample': [0.8, 1.0],
    'random_state': [42]
    }

    bst4 = tune_classification_model_hyperparameters(gradient_boosting_class, X_train, X_val, y_train, y_val, hyp3)

    save_model(bst4, 'gradient_boosting_classifier', 'models/classification/')


def find_best_model(main, task_folder):
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

def tune_classification_model_hyperparameters(model, X_train, X_val, y_train, y_val, hyperparameters):
    grid_search = GridSearchCV(model, hyperparameters, cv = 5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    y_val_pred = best_model.predict(X_val)
    validation_accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average = 'macro')
    recall = recall_score(y_val, y_val_pred, average = 'macro')
    f1 = f1_score(y_val, y_val_pred, average = 'macro')
    perf_metrics = {
        'Validation_Accuracy' : validation_accuracy,
        'Precision Score' : precision,
        'Recall Score' : recall,
        'F1 Score' : f1
    }
    return best_model, best_hyperparameters, perf_metrics

if __name__ == '__main__':
    # # Loads the data from the load_airbnb function and splits it into features and labels
    # X, y = load_airbnb('Price_Night')

    # # Splits the data into traning and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)

    # # Split data into training and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = 42)

    # # Evaluates all models and saves the model, hyperparameters and metrics in their respective folders
    # evaluate_all_models()

    # # Extracts, evaluates and returns the best model
    # best_model = find_best_model('models', 'regression')

    # # Prints best model
    # print(best_model)

    # Loads the data from the load_airbnb function and splits it into features and labels
    X,y = load_airbnb('Category')

    # Splits the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

    # model = LogisticRegression()
    # model.fit(X_train, y_train)

    # y_train_pred = model.predict(X_train)
    # train_accuracy = accuracy_score(y_train, y_train_pred)
    # train_precision = precision_score(y_train, y_train_pred, average = 'macro')
    # train_recall = recall_score(y_train, y_train_pred, average = 'macro')
    # train_f1 = f1_score(y_train, y_train_pred, average = 'macro')

    # y_test_pred = model.predict(X_test)
    # test_accuracy = accuracy_score(y_test, y_test_pred)
    # test_precision = precision_score(y_test, y_test_pred, average='macro')
    # test_recall = recall_score(y_test, y_test_pred, average='macro')
    # test_f1 = f1_score(y_test, y_test_pred, average='macro')

    # print("Training Accuracy:", train_accuracy)
    # print("Training Precision:", train_precision)
    # print("Training Recall:", train_recall)
    # print("Training F1 Score:", train_f1)
    # print("Test Accuracy:", test_accuracy)
    # print("Test Precision:", test_precision)
    # print("Test Recall:", test_recall)
    # print("Test F1 Score:", test_f1)

    bestclassmodel = find_best_model('models', 'classification')
    print(bestclassmodel)