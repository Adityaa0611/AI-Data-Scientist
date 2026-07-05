import pandas as pd

def clean_data(df):
    """
    This function acts as the Cleaning Agent.
    It takes the raw dataframe and applies automatic cleaning rules.
    """
    # Rule 1: Remove any exact duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Rule 2: Handle missing numbers
    # Finds all columns with numbers and fills missing blanks with the median value
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())
    
    # Rule 3: Handle missing text
    # Finds all text columns and fills missing blanks with the word 'Unknown'
    text_cols = df_cleaned.select_dtypes(include=['object']).columns
    df_cleaned[text_cols] = df_cleaned[text_cols].fillna('Unknown')
    
    return df_cleaned