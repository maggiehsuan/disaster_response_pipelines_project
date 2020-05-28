import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    '''
    Args:
        messages_filepath: File path of messages data
        categories_filepath: File path of categories data
    Returns:
        df: Merged dataset from messages and categories
    '''

    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on=['id'])

    return df


def clean_data(df):
    '''
    Args:
        df: Merged dataset from messages and categories
    Returns:
        df_cl: A complete cleaned dataset
    '''
    # create a dataframe of the 36 individual category columns
    category_cols = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = category_cols.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split("-",expand=True)[0]

    # rename the columns of `categories`
    category_cols.columns = category_colnames

    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    # related-0 becomes 0, related-1 becomes 1.
    for column in category_cols:
    # set each value to be the last character of the string
        category_cols[column] = category_cols[column].str.split("-").str.get(1)
    
    # convert column from string to numeric
    category_cols[column] = category_cols[column].astype('int')

    # drop the categories column from `df`
    messages = df.drop('categories', axis=1)

    # concatenate the dataframe with the cleaned category columns
    df_cl = pd.concat([messages, category_cols], axis=1)

    # drop duplicates
    df_cl = df_cl.drop_duplicates()

    return df_cl


def save_data(df, database_filename):
    '''
    Args:
        df: cleaned dataset
        database_filename: database name, e.g. DisasterResponse.db
    Returns: 
        A SQLite database
    '''

    # Save df into sqlite database
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('clean_table', con=engine, index=False,if_exists='replace')
  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df_cl = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df_cl, database_filepath)
        
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