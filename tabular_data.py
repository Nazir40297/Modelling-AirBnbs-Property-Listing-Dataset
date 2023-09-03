import pandas as pd 
import ast

def clean_tabular_data(df):
    '''
    Cleans a tabular DataFrame by removing rows with missing ratings.

    This function takes a DataFrame containing ratings data for various attributes like cleanliness, accuracy, communication, etc.
    It removes rows with missing values in any of these rating columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing ratings data.

    Returns:
    pd.DataFrame: A cleaned DataFrame with rows containing missing ratings removed.
    '''
    def remove_rows_with_missing_ratings(df):
        '''
         Removes rows with missing ratings.

        This function takes a DataFrame and removes rows with missing values in any of the specified rating columns.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing ratings data.

        Returns:
        pd.DataFrame: A DataFrame with rows containing missing ratings removed.
        '''
        ratings = ['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating']
        # Removing entries with missing values
        df = df.dropna(subset = ratings)
        return df
    
    df = remove_rows_with_missing_ratings(df)

    def combine_description_strings(df):
        '''
        Combines and processes description strings.

        This function takes a DataFrame and processes the 'Description' column by dropping rows with missing descriptions,
        removing a specific prefix, and converting string representations of lists into actual lists.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing descriptions.

        Returns:
        pd.DataFrame: A DataFrame with cleaned and processed descriptions.
        '''
        # Drop entries missing a description
        df = df.dropna(subset = 'Description')
        # Remove the "About this space" prefix from each description
        df['Description'] = df['Description'].str.replace('About this space', '')
        # Convert the string representations of lists into actual lists
        df['Description'] = df['Description'].apply(lambda x: ''.join(ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x))
        return df

    df = combine_description_strings(df)

    def set_default_feature_values(df):
        '''
        Sets default values for specific feature columns.

        This function takes a DataFrame and fills missing values with default values in specific feature columns.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing feature columns.

        Returns:
        pd.DataFrame: A DataFrame with missing feature values filled with defaults.
        '''
        list = ['guests', 'beds', 'bathrooms', 'bedrooms']
        # Fill missing values with entry 1
        df[list] = df[list].fillna(1)
        return df
    
    df = set_default_feature_values(df)

    # Drop 'Unnamed: 19' column
    df = df.drop('Unnamed: 19', axis = 1)
    # Drop row 586 due to mixed data along columns
    df = df.drop(index = 586)
    return df

def load_airbnb(label):
    '''
    Load and preprocess Airbnb data for machine learning.

    This function loads a CSV file containing Airbnb property listings data, performs data type conversions
    for specific columns, separates features and labels, and returns them as separate DataFrames.

    Parameters:
    label (str): The name of the column to be used as the target label.

    Returns:
    tuple: A tuple containing two DataFrames:
        - First DataFrame: Contains the selected features for machine learning.
        - Second DataFrame: Contains the target labels.

    Example:
    features, labels = load_airbnb('Price_Night')
    '''
    # Load the AirBnb data from the CSV file
    df = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
    # Convert selected columns to the appropriate data types
    df['guests'] = df['guests'].astype(float)
    df['Price_Night'] = df['Price_Night'].astype(float)
    df['bedrooms'] = df['bedrooms'].astype(int)
    # Select numerical columns for machine learning
    numerical_columns = df.select_dtypes(include=[int, float]).columns.tolist()
    df_numerical = df[numerical_columns]

    # Extract labels (target variable) and features
    labels = df[label]
    if label in df_numerical.columns:
        # If the label is numerical, drop it from features
        features = df_numerical.drop(columns=[label])
    else:
        # Use all numerical columns as features if the label is not numerical
        features = df_numerical
    return features, labels

if __name__ == '__main__':
    # Define the path to the input CSV file
    filepath = 'airbnb-property-listings/tabular_data/listing.csv'
    # Read the data from the CSV file into a DataFrame
    df = pd.read_csv(filepath)
    # Clean the tabular data by calling the clean_tabular_data function
    df = clean_tabular_data(df)
    # Save the cleaned DataFrame to the output CSV file (excluding the index)
    df.to_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv', index=False)

