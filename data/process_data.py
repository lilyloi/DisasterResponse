import sys
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories from dataset
    
    Args:
      message_filepath(string): the file path of messages.csv
      categories_filepath(string): the file path of categories.csv
      
    Return:
      df(Dataframe): merged dataframe of messages + categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')

def clean_data(df):
    """
    Clean up data:
      1. Drop duplicates
      2. Remove missing values
      3. Clean categories 
    Args: 
      df(Dataframe): merged dataframe of messages + categories from load_data()

    Return:
      df(Dataframe): cleaned dataframe 
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    
    # Get new column names from category columns
    category_colnames = row.apply(lambda x: x.rstrip('- 0 1'))
    categories.columns = category_colnames
    
    # Convert category values to 0 or 1
    categories = categories.applymap(lambda s: int(s[-1]))

    # Drop the original categories column from Dataframe
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Clean up to get final dataframe 
    df.drop_duplicates(subset='message', inplace=True)
    df.dropna(subset=category_colnames, inplace=True)
    df = df[df.related != 2]
    df = df.drop('child_alone', axis=1)
    
    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into SQLite database
    
    Args:
        df(Dataframe): the cleaned dataframe
        database_filename(string): the file path to save file .db
    Return:
        None

    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DS_messages', engine, index=False, if_exists='replace')
    engine.dispose()

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