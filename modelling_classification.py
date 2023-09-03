from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from modelling_regression import find_best_model, save_model, split_data
from tabular_data import load_airbnb


def evaluate_all_class_models():
    """
    Evaluate various classification models using hyperparameter tuning.

    This function evaluates classification models like DecisionTreeClassifier, RandomForestClassifier, and GradientBoostingClassifier.
    It calls the hyperparameter tuning functions for each model and saves the best models with their details.

    Note: This function does not use stratified k-fold cross-validation.

    Returns:
    None
    """
    # Create a DecisionTreeClassifier
    m1 = DecisionTreeClassifier()

    # Define hyperparameters for DecisionTreeClassifier
    hyp = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
    }

    # Tune hyperparameters for DecisionTreeClassifier
    bst2 = tune_classification_model_hyperparameters(m1, X_train, X_val, y_train, y_val, hyp)

    # Save the best DecisionTreeClassifier model
    save_model(bst2, 'decision_tree_classifier', 'models/classification/')

    # Create a RandomForestClassifier
    random_forest_class = RandomForestClassifier()

    # Define hyperparameters for RandomForestClassifier
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

    # Tune hyperparameters for RandomForestClassifier
    bst3 = tune_classification_model_hyperparameters(random_forest_class, X_train, X_val, y_train, y_val, hyp2)

    # Save the best RandomForestClassifier model
    save_model(bst3, 'random_forest_classifier', 'models/classification/')

    # Create a GradientBoostingClassifier
    gradient_boosting_class = GradientBoostingClassifier()

    # Define hyperparameters for GradientBoostingClassifier
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

    # Tune hyperparameters for GradientBoostingClassifier
    bst4 = tune_classification_model_hyperparameters(gradient_boosting_class, X_train, X_val, y_train, y_val, hyp3)

    # Save the best GradientBoostingClassifier model
    save_model(bst4, 'gradient_boosting_classifier', 'models/classification/')

def evaluate_all_class_models_strat():
    """
    Evaluate various classification models using stratified k-fold cross-validation.

    This function is similar to evaluate_all_class_models but uses stratified k-fold cross-validation
    for classification models. It also saves the best models with their details.

    Returns:
    None
    """
    # Create a DecisionTreeClassifier with class_weight 'balanced'
    m1 = DecisionTreeClassifier(class_weight = 'balanced')

    # Define hyperparameters for DecisionTreeClassifier
    hyp = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced'],
    'random_state': [42]
    }

    # Tune hyperparameters for DecisionTreeClassifier using stratified k-fold cross-validation
    bst2 = tune_classification_model_hyperparameters_strat(m1, X_train, X_val, y_train_comb, y_val_comb, hyp)

    # Save the best DecisionTreeClassifier model
    save_model(bst2, 'decision_tree_classifier_bedroom', 'models/classification/')

    # Create a RandomForestClassifier with class_weight 'balanced'
    random_forest_class = RandomForestClassifier(class_weight = 'balanced')

    # Define hyperparameters for RandomForestClassifier
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

    # Tune hyperparameters for RandomForestClassifier using stratified k-fold cross-validation
    bst3 = tune_classification_model_hyperparameters_strat(random_forest_class, X_train, X_val, y_train_comb, y_val_comb, hyp2)

    # Save the best RandomForestClassifier model
    save_model(bst3, 'random_forest_classifier_bedroom', 'models/classification/')

    # Create a GradientBoostingClassifier
    gradient_boosting_class = GradientBoostingClassifier()

    # Define hyperparameters for GradientBoostingClassifier
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

    # Tune hyperparameters for GradientBoostingClassifier using stratified k-fold cross-validation
    bst4 = tune_classification_model_hyperparameters_strat(gradient_boosting_class, X_train, X_val, y_train_comb, y_val_comb, hyp3)

    # Save the best GradientBoostingClassifier model
    save_model(bst4, 'gradient_boosting_classifier_bedroom', 'models/classification/')

def tune_classification_model_hyperparameters(model, X_train, X_val, y_train, y_val, hyperparameters):
    '''
    Tune hyperparameters for a classification model using GridSearchCV and evaluate its performance.

    This function performs hyperparameter tuning for a given classification model using GridSearchCV.
    It takes training and validation data along with a hyperparameter grid and evaluates the model's
    performance on the validation set.

    Parameters:
        model (sklearn.base.BaseEstimator): The classification model to be tuned.
        X_train (array-like): Training data features.
        X_val (array-like): Validation data features.
        y_train (array-like): Training data labels.
        y_val (array-like): Validation data labels.
        hyperparameters (dict): The hyperparameter grid to search over.

    Returns:
        tuple: A tuple containing the best model, best hyperparameters, and performance metrics.

    Performance Metrics:
        - Validation_Accuracy: Accuracy on the validation set.
        - Precision Score: Precision score on the validation set (macro average).
        - Recall Score: Recall score on the validation set (macro average).
        - F1 Score: F1 score on the validation set (macro average).
    '''
    # Use GridSearchCV to perform hyperparameter tuning
    grid_search = GridSearchCV(model, hyperparameters, cv = 5)
    grid_search.fit(X_train, y_train)

    # Use GridSearchCV to perform hyperparameter tuning
    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Predict on the validation set and calculate performance metrics
    y_val_pred = best_model.predict(X_val)
    validation_accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average = 'macro', zero_division=1)
    recall = recall_score(y_val, y_val_pred, average = 'macro', zero_division=1)
    f1 = f1_score(y_val, y_val_pred, average = 'macro', zero_division=1)

    # Performance metrics dictionary
    perf_metrics = {
        'Validation_Accuracy' : validation_accuracy,
        'Precision Score' : precision,
        'Recall Score' : recall,
        'F1 Score' : f1
    }
    # Return the best model, best hyperparameters, and performance metrics
    return best_model, best_hyperparameters, perf_metrics

def tune_classification_model_hyperparameters_strat(model, X_train, X_val, y_train, y_val, hyperparameters):
    '''
    Tune hyperparameters for a classification model using Stratified K-Fold Cross-Validation and evaluate its performance.

    This function performs hyperparameter tuning for a given classification model using Stratified K-Fold Cross-Validation.
    It takes training and validation data along with a hyperparameter grid and evaluates the model's performance on the
    validation set.

    Parameters:
        model (sklearn.base.BaseEstimator): The classification model to be tuned.
        X_train (array-like): Training data features.
        X_val (array-like): Validation data features.
        y_train (array-like): Training data labels.
        y_val (array-like): Validation data labels.
        hyperparameters (dict): The hyperparameter grid to search over.

    Returns:
        tuple: A tuple containing the best model, best hyperparameters, and performance metrics.

    Performance Metrics:
        - Validation_Accuracy: Accuracy on the validation set.
        - Precision Score: Precision score on the validation set (macro average).
        - Recall Score: Recall score on the validation set (macro average).
        - F1 Score: F1 score on the validation set (macro average).
    '''
    # Create Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits = 4, shuffle = True, random_state = 42)

    # Use GridSearchCV with Stratified K-Fold cross-validation to perform hyperparameter tuning
    grid_search = GridSearchCV(model, hyperparameters, cv = cv)
    grid_search.fit(X_train, y_train)

    # Get the best model and its hyperparameters
    best_model = grid_search.best_estimator_
    best_hyperparameters = grid_search.best_params_

    # Predict on the validation set and calculate performance metrics
    y_val_pred = best_model.predict(X_val)
    validation_accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, average = 'macro', zero_division=1)
    recall = recall_score(y_val, y_val_pred, average = 'macro', zero_division=1)
    f1 = f1_score(y_val, y_val_pred, average = 'macro', zero_division=1)

    # Performance metrics dictionary
    perf_metrics = {
        'Validation_Accuracy' : validation_accuracy,
        'Precision Score' : precision,
        'Recall Score' : recall,
        'F1 Score' : f1
    }
    # Return the best model, best hyperparameters, and performance metrics
    return best_model, best_hyperparameters, perf_metrics

def combine_classes(y):
    '''
    Combine classes in a classification task to simplify the problem.

    This function is used to merge multiple classes into a single class in a classification task.
    It can be particularly helpful for simplifying complex classification problems with many classes.

    Parameters:
        y (array-like): The original class labels.

    Returns:
        array-like: An array with merged class labels.
    '''
    # Create a copy of the original class labels
    y_combined = y.copy()

    # Merge specified classes into a single class (e.g., 6, 7, 8, 10, and 5 into 4)
    y_combined[y_combined.isin([6, 7, 8, 10, 5])] = 4

    return y_combined

if __name__ == '__main__':
    # Loads the data from the load_airbnb function and splits it into features and labels
    X,y = load_airbnb('Category')

    # Split the data into training, validation and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Evaluates all classification models
    evaluate_all_class_models()

    # Finds best classification model based on Validation Accuracy
    bestclassmodel = find_best_model('models', 'classification')

    # Prints the best classification model
    print(bestclassmodel)

    X, y = load_airbnb('bedrooms')

    # Split the data into training, validation and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    y_train_comb = combine_classes(y_train)
    y_val_comb = combine_classes(y_val)
    y_test_comb = combine_classes(y_test)

    evaluate_all_class_models_strat()