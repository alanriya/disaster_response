import sys
import pdb
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Summary:
        read 2 csv and combine them into 1.
        
    Input:
        messages_filepath[str] : path in string to the messages csv 
        categories_filepath[str] : path in string to the categories filepath 
       
    Output:
        df : merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='inner')
    return df


def clean_data(df):
    """
    Summary:
        Cleaning data into usable format for the Machine learning Classifier.
        
    Input:
        df: merged df.
      
    Output:
        df : cleaned dataframe to be used in classifier model.
    """
    categories = df['categories'].str.split(pat=';', expand=True)
    # take the first row
    row = categories.iloc[0]
    # get column name for the first row.
    category_colnames = [i.split('-')[0] for i in row]
    # rename
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [i.split('-')[1] for i in categories[column]]
        # convert column from string to numeric
        categories[column] = categories[column].astype(float)
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.merge(df, categories, left_index=True, right_index=True)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Summary:
        Create sqlite engine and store the dataframe to database.
        
    Input:
        df: cleaned df.
        database_filename: file path in string to the database.
    """
    engine = create_engine('sqlite:///data/DisasterResponse.db')  
    df.to_sql('message',  con = engine, if_exists='append', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()