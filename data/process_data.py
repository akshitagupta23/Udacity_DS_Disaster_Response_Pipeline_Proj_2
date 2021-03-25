#! /usr/bin/env python3
# coding=utf-8

'''
Title: Disaster Response Project: ETL Pipeline
Author: Akshita Gupta

'''

import sys
import pandas as pd
import logging
from sqlalchemy import create_engine

logging.basicConfig(filename='logs/disaster_response_etl.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Function that loads and merge the message and categories files 
    to return the new dataframe as output
    
    Input: 
           Messages_filepath: messages file path
           categories: the categories dataset filepath
    Output: Pandas dataframe of the merged csv files
    """
    messages = pd.read_csv(messages_filepath, engine ='c')
    categories = pd.read_csv(categories_filepath, engine ='c')

    df = messages.merge(categories, how='inner', on= 'id')
    return df
    

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
        Fucntion perform cleaning operations like
        seperate columns from multiple categories, 
        extract categories values and deduplication
        Input: DataFrame
        Output: Cleaned dataframe
    """
    
    categories = df.categories.str.split(';', expand=True)
    
    # Defining category names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # Converting category types as int type
    for column in categories:
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    # Removing entry that is non-binary
    df = df[df['related'] != 2]
       
    return df

def save_data(df: pd.DataFrame, database_filename: str):
    """ 
    Persists the cleaned dataframe into  database 
    Input: 
        df: cleaned dataframe
        database_filename: database to store the cleaned dataframe 
    
    Output: None
    
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        logging.info('Loading and Merging Data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        logging.info('Cleaning the data...')
        df = clean_data(df)
        
        logging.info('Saving the data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        logging.info('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()