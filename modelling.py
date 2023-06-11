import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score
import numpy as np 
from tabular_data import load_airbnb

# Loads the data from the load_airbnb function and splits into features and labels
X, y = load_airbnb('Price_Night')

# Splits the data into traning and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Create and train the linear regression model
model = SGDRegressor()
model.fit(X_train, y_train)

# Predict the variable
y_pred = model.predict(X_test)

# Calculate the MSE
mse = MSE(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print("R-squared (R²):", r2)

def custom_tune_regression_model_hyperparameters():
    return best_model, best_hyperparameters, performance_metrics