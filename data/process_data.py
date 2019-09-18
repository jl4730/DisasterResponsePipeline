import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function will read in two datasets, then merge them by id
    Input: file address
    Output: merged dataset
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories,on='id')
    
    return df
    
def clean_data(df):
    '''
    This function will help seperate the columns in df.category by spliting it into 
    36 categories and converting the values to 1/0.
    
    In addition this function will drop the duplicates.
    
    Input: dataframe df
    Output: cleaned df with duplicates removed and categories seperated
    '''

    category_expand = df.categories.str.split(';',expand=True)  # split the original categories field
    category_expand.columns = category_expand.loc[0,:].apply(lambda x:x[:-2])  # change columns names
    
    for i in category_expand.columns:
        category_expand[i] = pd.to_numeric(category_expand[i].str[-1])  # convert the values to numeric
        
    df = df.assign(**category_expand)  # add the new columns into the original df

    df = df.drop('categories',axis=1)  # drop the old category
    
    df = df.drop_duplicates()  # remove duplicates
    
    return df

def save_data(df, database_filename):
    
    engine = create_engine('sqlite:///'+database_filename, echo=False)
    df.to_sql('DisasterResponseData', con=engine,if_exists='replace')

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