library(dplyr)

# Define a function to clean tabular data
clean_tabular_data <- function(df) {
  # Remove row 587
  df <- df %>% slice(-587)
  # Define a function to remove rows with missing ratings
  remove_rows_with_missing_ratings <- function(df) {
    # Specify the rating columns
    ratings <- c('Cleanliness_rating', 'Accuracy_rating', 'Communication_rating',
                 'Location_rating', 'Check.in_rating', 'Value_rating')
    # Remove rows with missing ratings
    df <- df %>% drop_na(Cleanliness_rating, Accuracy_rating, Communication_rating,
                         Location_rating, Check.in_rating, Value_rating)
    return(df)
  }
  df <- remove_rows_with_missing_ratings(df)
  
  # Define a function to combine and process description strings
  combine_description_strings <- function(df) {
    # Drop rows with missing descriptions
    df <- df %>% drop_na(Description)
    # Remove the "About this space" prefix from descriptions
    df$Description <- gsub('About this space', '', df$Description)
    # Convert string representations of lists into actual lists
    df$Description <- gsub("^\\[|\\]$", "", df$Description)
    return(df)
  }
  df <- combine_description_strings(df)
  
  # Define a function to set default feature values
  set_default_feature_values <- function(df) {
    # Change datatype of bedroom to integer
    df$bedrooms <- as.numeric(df$bedrooms)
    # Change datatype of guests to integer
    df$guests <- as.numeric(df$guests)
    # Specify the list of feature columns
    features <- c('guests', 'beds', 'bathrooms', 'bedrooms')
    # Fill missing values with 1
    df <- df %>% mutate(across(all_of(features), ~ifelse(is.na(.), 1, .)))
    return(df)
  }
  df <- set_default_feature_values(df)
  
  # Drop the 'X' column
  df <- df %>% select(-X)
  
  return(df)
}

# Importing dataset
data = read.csv('airbnb-property-listings/tabular_data/listing.csv')
# Clean the data
clean_data <- clean_tabular_data(data)
# Filepath to save the CSV file
filepath <- 'airbnb-property-listings/tabular_data/cleaned_data.csv'
# Save the cleaned data to a CSV file
write.csv(clean_data, file = filepath, row.names = FALSE)
