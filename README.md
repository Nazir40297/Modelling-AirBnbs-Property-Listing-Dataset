# modelling-airbnbs-property-listing-dataset-

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Summary of functions](#functions)
- [License](#license)

## Description

This project involves training and evaluating various machine learning models for predicting Airbnb nightly prices. It covers both regression and classification tasks, aiming to achieve accurate predictions and understand feature importance.

The project involves the following key steps:
- Data loading and preprocessing using `load_airbnb` function.
- Data splitting into training, validation, and test sets.
- Training and tuning various regression and classification models.
- Evaluation of model performance using metrics like RMSE, R^2, accuracy, precision, recall, and F1-score.
- Neural network training and evaluation using PyTorch.

Through this project, we aim to gain insights into selecting and tuning machine learning models effectively for predicting Airbnb nightly prices.

## Installation

Clone the repository.
Install the required Python packages using the 'environment.yml' file. Create and activate the environment.
Download and load the dataset using the 'load_airbnb' function in the 'tabular_data.py' file.

## Usage

Modify the dataset loading process in the code as needed.
Run the provided code in the appropriate sections to evaluate regression and classification models as well as neural networks.
Modify the hyperparameter search spaces and other configurations as per your experiment requirements.

## Functions

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

We also created a function called 'load_airbnb' which takes in the label we want to predict. We loaded the cleaned tabular data and casted the 'guests' and 'bedrooms' columns to float to use in our features.
This function returns a tuple of the features and label and which we will use later for training our models. 

We created a file called modelling.py which contains our code used for modelling different regression models and training it on the data. We started with importing the load_airbnb function outlined above. We started with establishing a baseline model to improve upon. We used SGDRegressor to build our model and computed the key measures of performance namely, RMSE (root mean squared error) and R^2 values. 

Now the interesting part which we did rather than just using the traditional methods provided to us by SKLearn we built a custom model to tune the hyperparameters using GridSearch. So we wanted this function to take in a class, training set, validation set and a dictionary of hyperparameters which would be iterated over finding the best hyperparameter values. The function would return the best model, a dictionary of the best hyperparameters and a dictionary of the best perofrmance metrics. We started with empty dictionaries. Quite straightforward we iterated through the different values in the hyperparameters dictionary using a traditonal loop and fit the models with the corresponding parameters. The metrics were calculated and each model's RMSE was compared to the last. If the RMSE of the current model is lower than the previous model then the current model would replace the previous one in all relevant dictionaries. 

We then moved on to using the SKLearn method GridSearchCV to get a more accurate representation of our data due to the error being way too high. This time quite straighforward again, we passed a parameter grid and set scoring to maximise the negative root mean squared error. The function then would take the parameter grid and perform a search for the best hyperparameter values. 

We continued to apply this to various different models and saved the models in their respective folders. The models which were used are DecisionTreeRegressor, RandomForestRegressor and GradientBoostingRegressor. A function called find_the_best_model was made to extract the relevant model, hyperparameters and metrics; the metrics were then evaluated using the best RMSE score to find the best model. The model with the lowest RMSE score would be returned from the function alongside the model it is and the corresponding hyperparameter values.

For further experiment and attemots to lower the error of the models we train would be to try other hyperparameter tuning methods besides just GridSearchCV. 

We imported the function 'load_airbnb'. This time the label we wanted to predict was the 'Category'. We first trained a logisitic regression model to begin with. Just like above. A list of key measures of performance were then calculated to see how this logistic regression model performs. The measures of the performace were F1 score, precision, recall, accuracy scores for the training and test sets. This was established as the baseline model which we worked to improve on. 

Another function called 'tune_classification_hyperparameters' was made which is a lot similar to the earlier "tune hyperparameters" function used in the last model however this used the validation_accuracy to decide which hyperparameters were best for the model from a dictionary of list of hyperparameters. The classification counterparts of the regression models used previously were then trained and saved in their respective folders.

To further improve the results found from training these specific classification model would be to apply different types of hyperparameter tuning methods. Some methods work better on some data than others.

This part of the project we configured a neural network to predict the nightly listing price from the numerical data as the features. Mostly to configure this neural network we used PyTorch.

To begin with we created our PyTorch dataset which would return a tuple of the features and labels. The features were made as tensors representing the numerical tabular features of a house. Dataloader was then created to shuffle the relevant train and test sets. 

Then we defined a pytorch model class which would contain the architecture of our neural network. Each part of the network will be described and how it contributes to the overall model. 

```python
layers = []
prev_size = input_size
```
'layers': this is an empty list where the layers of the neural network will be stored as they are defined.
'prev_size': this variable is used to keep track of the size of the neurons of the previous layer. It starts with the input size which is the number of features in the input data.

```python
for size in self.config['hidden_layer_width']
```
'self.config['hidden_layer_width']': this references a dictionary named 'config' which will contain the hyperparameters for the network.
'hidden_layer_width': this will be a list representing the number of neurons in each layer.

```python
layers.append(nn.Linear(prev_size, size))
layers.append(nn.ReLU())
prev_size = size
```
This part of the code creates the hidden layers in the neural network. For each value size in the hidden_layer_width it does the following:

'nn.Linear(prev_size, size)': this creates a fully connected layer that connects the previous layer with neurons to the current layer with 'size' neurons. This defines the weight matrix and the bias vector to be learned during training.

'nn.ReLU()': this adds a rectified layer unit activation function after each linear layer. ReLU is an activation function that introduces non-linearity to the network.

'prev_size = size': this updates the 'prev_size' to the current 'size'. The current layer will become the previous layer for the next iteration.

```python
layers.append(nn.Linear(prev_size, 1))
self.layers = nn.Sequential(*layers)
```
The first line adds the final linear layer that connects the last hidden layer to a single neuron output layer. 
The second line in the above code creates a sequential container that stacks all the layers defined in the layers list. A sequential container allows you to define a neural network as a sequence of layers. The '*layers' syntax unpacks the list into individual arguments, building the neural network architecture.

In summary, this code defines the architecture of a neural network with multiple hidden layers, each followed by a ReLU acitvation function. Then a final linear output layer for regression use.

We then went on to define the forward function.

```python
def forward(self, x):
        return self.layers(x)
```
'def forward(self, x):' : this defines the forward method of the neural network class. This method takes two parameters: 'self' which is an instance of the class and 'x' which is the input data.

'return self.layers(x)': this passes the input data 'x' through the layers of the neural network using the 'self.layers' container that was defined earlier. 

The 'self.layers' container was created using the 'nn.Sequential' class and it holds the layers of the network in the order defined. When the input data is passed through the layers, PyTorch automatically performs the forward pass through each layer in the sequence applying the weights, biases and activation functions.

In summary, the 'forward' method specifies how input data should flow through the network's layer to produce predictions. When 'model(input_data)' is called during inference, this forward method is automatically executed, applying the transformations defined in the network's architecture to generate the output.

We then defined our train function which would be the main fucntion to train the neural network. 

```python
criterion = nn.MSELoss()
optimiser = torch.optim.__dict__[config['optimiser']](model.parameters(), lr = config['learning_rate'])
```
Loss and optimiser: the loss criterion measures the difference between predicted values and actual labels. The optimiser adjusts the model parameters to minimise this loss. The optimiser is selected based on the configuration provided in the 'config' dictionary.

```python
for epoch in range(num_epochs):
        model.train()
```
Training loop: for each epoch the model is set to training mode ('model.train()'). This is important for layers like dropout that behave differently during training and evaluation.

```python
for batch_features, batch_labels in train_dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimiser.zero_grad()
```
Batch iteration: the training data is iterated over in batches using the 'train_dataloader'.
Moving to device: the batch features and labels are moved to the specified device to take advantage of hardware acceleration.
Zero gradients: gradients which are accumulated from the previous iteration are cleared using 'optimiser.zero_grad()'.

```python
output = model(batch_features)
```
Forward pass: the model's forward pass computes predicted outputs based on the input features. 

```python
loss = criterion(output, batch_labels)
loss.backward()
optimiser.step()
```
Loss computation: the loss is calculated by comparing the predicted output to the actual batch labels.
Backpropogation and optimisation: gradients are computed through backpropagation 'loss.backward()' and the optimiser updates the model's parameters based on these gradients using 'optimiser.step()'.

```python
writer.add_scalar('Loss/train', loss.item(), epoch)
```
Training loss logging: the training loss is logged to a visualisation tool to moniter how the loss decreases over epochs. 

```python
model.eval()
        with torch.no_grad():
            for val_features, val_labels in validation_dataloader:
                val_output = model(val_features)
                val_loss = criterion(val_output, val_labels)
                validation_loss = val_loss.item()
            
                writer.add_scalar('Loss/validation', validation_loss, epoch)
```
Model evaluation mode: the model is switched to evaluation mode to disable certain operations that should only be active during training.
'torch.no_grad()' makes sure all the operations performed using PyTorch will not record the gradients. This prevents unneccessary memory usage.
Validation looping and loss calculation: for each batch of validation data from the validation dataloader the model computes predicted outputs. The validation loss is calculated by comparing the predicted outputs with the actual labels using the same loss criterion as during training. 
Logging validation loss: the calculated validation loss is logged using a writer for visualisation. 

```python
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
```
Training duration calculation: after the training loop, the end time is recorded and then the total training duration is calculated as the difference between the times. 
Model evalution mode: the model is set to evaluation mode again before inference latency calculation. 
Inference latency calculation: a new timer is started and then the validation data is iterated through again. However, this time only measures the time taken for model inference only. After the inference loop the end time is recorded. The total inference duration is calculated as a difference between the two times. The inference latency is calculated by dividing the total inference duration by the product of the number of validation batches by the number of samples in each batch.

