# modelling-airbnbs-property-listing-dataset-

Milestone 1

We loaded the data from a CSV file using pandas. Before we start to use the data it is essential we clean the data and prepare it for moelling. We created a number of functions to clean the data...

We started off with a function called 'remove_rows_with_missing_ratings' which takes in a dataframe and removed all the properties on the AirBnb dataset which did not have any ratings.

We then created a function called 'combine_description_strings'. In one of the columns of our dataframe we have a string of lists. Pandas reads these lists as strings so we needed to combine these lists into 
one string. First off we removed all the entries which had no description, then removed the "About this space" from each entry and replacing it with whitespace. Then using list comprehension and namely the 
'join()' and 'ast_literal_eval' functions we combined the lists of strings all into one string. A problem did occur whereby one of the entries was just a string and not a list where we kept receiving errors. 
We modify the line of code to take into consideration this instance. So if it was a string it would be left alone and only the entries which began with "[" would be edited. This returned us a neat column of
description for each property.

We then went on to create a function called 'set_default_feature_values' which instead of removing the entries with null values filled each null value with a rating of 1. Yes, not a very good rating but rather
than omitting this data from the dataset it would be better for us to include it.

When loading the data we were presented with an unwanted column which we have dropped and the entry which we had problems with upon inspection had mixed data across the columns so we omitted this also. These
3 functions were then put under one function called 'clean_tabular_data' for easy usage. 

Under the if __name__ == '__main__' we load the file clean it using the above functon and then save it back into our folder.

Milestone 2

We also created a function called 'load_airbnb' which takes in the label we want to predict. We loaded the cleaned tabular data and casted the 'guests' and 'bedrooms' columns to float to use in our features.
This function returns a tuple of the features and label and which we will use later for training our models. 

We created a file called modelling.py which contains our code used for modelling different regression models and training it on the data. We started with importing the load_airbnb function outlined above. We started with establishing a baseline model to improve upon. We used SGDRegressor to build our model and computed the key measures of performance namely, RMSE (root mean squared error) and R^2 values. 

Now the interesting part which we did rather than just using the traditional methods provided to us by SKLearn we built a custom model to tune the hyperparameters using GridSearch. So we wanted this function to take in a class, training set, validation set and a dictionary of hyperparameters which would be iterated over finding the best hyperparameter values. The function would return the best model, a dictionary of the best hyperparameters and a dictionary of the best perofrmance metrics. We started with empty dictionaries. Quite straightforward we iterated through the different values in the hyperparameters dictionary using a traditonal loop and fit the models with the corresponding parameters. The metrics were calculated and each model's RMSE was compared to the last. If the RMSE of the current model is lower than the previous model then the current model would replace the previous one in all relevant dictionaries. 

We then moved on to using the SKLearn method GridSearchCV to get a more accurate representation of our data due to the error being way too high. This time quite straighforward again, we passed a parameter grid and set scoring to maximise the negative root mean squared error. The function then would take the parameter grid and perform a search for the best hyperparameter values. 

We continued to apply this to various different models and saved the models in their respective folders. The models which were used are DecisionTreeRegressor, RandomForestRegressor and GradientBoostingRegressor. A function called find_the_best_model was made to extract the relevant model, hyperparameters and metrics; the metrics were then evaluated using the best RMSE score to find the best model. The model with the lowest RMSE score would be returned from the function alongside the model it is and the corresponding hyperparameter values.

For further experiment and attemots to lower the error of the models we train would be to try other hyperparameter tuning methods besides just GridSearchCV. 



