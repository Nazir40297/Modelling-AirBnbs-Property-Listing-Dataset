import pandas as pd 
import ast

def clean_tabular_data(df):
    def remove_rows_with_missing_ratings(df):
        ratings = ['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating']
        # removing entries with missing values
        df = df.dropna(subset = ratings)
        return df
    
    df = remove_rows_with_missing_ratings(df)

    def combine_description_strings(df):
        # drop entries missing a description
        df = df.dropna(subset = 'Description')
        # Remove the "About this space" prefix from each description
        df['Description'] = df['Description'].str.replace('About this space', '')
        # Convert the string representations of lists into actual lists
        df['Description'] = df['Description'].apply(lambda x: ''.join(ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x))
        return df

    df = combine_description_strings(df)

    def set_default_feature_values(df):
        list = ['guests', 'beds', 'bathrooms', 'bedrooms']
        # fill missing values with entry 1
        df[list] = df[list].fillna(1)
        return df
    
    df = set_default_feature_values(df)

    # drop 'Unnamed: 19' column
    df = df.drop('Unnamed: 19', axis = 1)
    # drop row 586 due to mixed data along columns
    df = df.drop(index = 586)
    return df

def load_airbnb(label):
    df = pd.read_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv')
    df['guests'] = df['guests'].astype(float)
    df['bedrooms'] = df['bedrooms'].astype(float)
    numerical_columns = df.select_dtypes(include=[int, float]).columns.tolist()
    df_numerical = df[numerical_columns]

    labels = df[label]
    if label in df_numerical.columns:
        features = df_numerical.drop(columns=[label])
    else:
        features = df_numerical
    return features, labels

if __name__ == '__main__':
    filepath = 'airbnb-property-listings/tabular_data/listing.csv'
    df = pd.read_csv(filepath)
    df = clean_tabular_data(df)
    df.to_csv('airbnb-property-listings/tabular_data/clean_tabular_data.csv', index=False)

