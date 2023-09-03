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
from modelling_regression import save_model
from modelling_regression import find_best_model

def evaluate_all_class_models():
    '''This function evaluates various classification models using hyperparameter tuning. 
    It evaluates models like DecisionTreeClassifier, RandomForestClassifier, and GradientBoostingClassifier. 
    It uses accuracy, precision, recall, and F1-score for evaluation.'''
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
    'max_features': ['sqrt'],
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

def evaluate_all_class_models_strat():
    '''This function is similar to the previous one but uses stratified k-fold cross-validation for classification models.'''
    m1 = DecisionTreeClassifier(class_weight = 'balanced')

    hyp = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
    }

    bst2 = tune_classification_model_hyperparameters_strat(m1, X_train, X_val, y_train_comb, y_val_comb, hyp)

    save_model(bst2, 'decision_tree_classifier_bedroom', 'models/classification/')

    random_forest_class = RandomForestClassifier(class_weight = 'balanced')

    hyp2 = {
    'n_estimators': [100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt'],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
    }

    bst3 = tune_classification_model_hyperparameters_strat(random_forest_class, X_train, X_val, y_train_comb, y_val_comb, hyp2)

    save_model(bst3, 'random_forest_classifier_bedroom', 'models/classification/')

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

    bst4 = tune_classification_model_hyperparameters_strat(gradient_boosting_class, X_train, X_val, y_train_comb, y_val_comb, hyp3)

    save_model(bst4, 'gradient_boosting_classifier_bedroom', 'models/classification/')

def tune_classification_model_hyperparameters(model, X_train, X_val, y_train, y_val, hyperparameters):
    grid_search = GridSearchCV(model, hyperparameters, cv = 5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    y_val_pred = best_model.predict(X_val)
    validation_accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average = 'macro', zero_division=1)
    recall = recall_score(y_val, y_val_pred, average = 'macro', zero_division=1)
    f1 = f1_score(y_val, y_val_pred, average = 'macro', zero_division=1)
    perf_metrics = {
        'Validation_Accuracy' : validation_accuracy,
        'Precision Score' : precision,
        'Recall Score' : recall,
        'F1 Score' : f1
    }
    return best_model, best_hyperparameters, perf_metrics

def tune_classification_model_hyperparameters_strat(model, X_train, X_val, y_train, y_val, hyperparameters):
    cv = StratifiedKFold(n_splits = 4, shuffle = True, random_state = 42)

    grid_search = GridSearchCV(model, hyperparameters, cv = cv)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    y_val_pred = best_model.predict(X_val)
    validation_accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average = 'macro', zero_division=1)
    recall = recall_score(y_val, y_val_pred, average = 'macro', zero_division=1)
    f1 = f1_score(y_val, y_val_pred, average = 'macro', zero_division=1)
    perf_metrics = {
        'Validation_Accuracy' : validation_accuracy,
        'Precision Score' : precision,
        'Recall Score' : recall,
        'F1 Score' : f1
    }
    return best_model, best_hyperparameters, perf_metrics

def combine_classes(y):
    '''This function is used to combine classes in classification tasks. 
    It merges multiple classes into a single class, typically used for simplifying classification tasks.'''
    y_combined = y.copy()
    y_combined[y_combined.isin([6, 7, 8, 10, 5])] = 4

    return y_combined

if __name__ == '__main__':
    # Loads the data from the load_airbnb function and splits it into features and labels
    X,y = load_airbnb('Category')

    # Splits the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average = 'macro')
    train_recall = recall_score(y_train, y_train_pred, average = 'macro')
    train_f1 = f1_score(y_train, y_train_pred, average = 'macro')

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    print("Training Accuracy:", train_accuracy)
    print("Training Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F1 Score:", train_f1)
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1 Score:", test_f1)

    # Evaluates all classification models
    evaluate_all_class_models()

    # Finds best classification model based on Validation Accuracy
    bestclassmodel = find_best_model('models', 'classification')

    # Prints the best classification model
    print(bestclassmodel)

    X, y = load_airbnb('bedrooms')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = 42)

    y_train_comb = combine_classes(y_train)
    y_val_comb = combine_classes(y_val)
    y_test_comb = combine_classes(y_test)

    model = LogisticRegression()
    model.fit(X_train, y_train_comb)

    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train_comb, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average = 'macro', zero_division=1)
    train_recall = recall_score(y_train, y_train_pred, average = 'macro', zero_division=1)
    train_f1 = f1_score(y_train, y_train_pred, average = 'macro', zero_division=1)

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test_comb, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=1)
    test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=1)
    test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=1)

    print("Training Accuracy:", train_accuracy)
    print("Training Precision:", train_precision)
    print("Training Recall:", train_recall)
    print("Training F1 Score:", train_f1)
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1 Score:", test_f1)

    evaluate_all_class_models_strat()