import sys
import pandas as pd
import numpy as np
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    Usage: read and merge messages and categories csv files into one dataframe objects

    Args:
        messages_filepath (str): filepath and filename for messages file 
        categories_filepath (str): filepath and filename for categories file 
    
    Returns:
        the merged dataframe
    '''
    # read messages file
    print('Reading message dataset.')
    df_messages = pd.read_csv(messages_filepath, delimiter = ',')
    # read categories file
    print('Reading category dataset.')
    df_categories = pd.read_csv(categories_filepath, delimiter = ',')
    # merge datasets
    print('Merging two datasets.')
    df_merge = df_messages.merge(df_categories, how = 'outer', on = 'id')
    return df_merge

def deduplicate(df):
    '''
    Usage: remove all duplicate values in the dataframe
        
    Args:
        df (dataframe): the dataframe you want to perform deduplication
           
    Returns:
        None
    '''
    # check number of duplicates
    df_duplicated = df.duplicated()
    n_duplicates = df[df_duplicated == True].shape[0]
    print("{} duplicates found and removed from the dataset".format(n_duplicates))
    # drop duplicates
    if n_duplicates > 0:
        df = df[~df_duplicated]
    return

# sanity check on the values in each column
def sanity_check(df):
    '''
    Usage: Check if there is any value other than 0 or 1 for each column.
    '''
    flag = False
    for col in df.columns:
        value_set = set(df[col].unique())
        if (value_set - set([0, 1])):
            flag = True
            print("The column '{}' contains additional values: {}".format(col,value_set))
            idx = (~df[col].isin([0, 1]))
            print("There are {} out of {} ({}%) in the 'related' column" \
              .format(df[idx].shape[0], df.shape[0], 
                      df[idx].shape[0]/df.shape[0]*100))
            #print("Impute the abnormal value with the mode {} in this column.".format(df[col].mode()[0]))
            #df.loc[idx, col] = df[col].mode()[0]
    if flag is False:
        print("All data entries are either 0 or 1 in the dataset")
    return

def clean_data(df):
    '''
    Usage: merge two datasets of messages categories, and transform into one dataframe with customized format
        
    Args:
        df (dataframe): the dataframe to apply cleaning processes
        
    Returns:
        a cleaned dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(data = np.zeros((df.shape[0], 36), dtype = float))
    # split categories into separate category columns
    category_colnames = df.iloc[0] \
                        .str.split(';', expand = True).loc['categories'].apply(lambda x : x.split('-')[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    print('Parsing category values row-wise.')
    # convert category values row-wise
    for irow, row in enumerate(df.iterrows()):
        # list of categories in each row of df.categories
        cat_list = row[1][4].split(';')
        # only keep the last letter, which is "1" or "0"
        cat_list = [x[-1] for x in cat_list]
        # convert from list to integer array
        cat_array = np.array(cat_list).astype(float)
        # set array values to categories dataframe by row index
        categories.iloc[irow,:] = cat_array
    # impute values that are not 0 or 1 with the mode in the corresponding column
    sanity_check(categories)
    # drop categories that contain 100% of 1s or 0s
    cat1 = categories.sum()/categories.shape[0]
    cat0 = 1 - cat1
    col_todrop = cat1[(cat1 == 0) | (cat1 == 1)].index.values.tolist()
    for col in col_todrop:
        print("drop categories that contain 100% of 1s or 0s: {}".format(col))
    categories.drop(labels = col_todrop, axis = 1, inplace = True)
    # drop the original categories column from `df`
    df.drop(labels = 'categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    print('Removing duplicated values.')
    # duplication
    deduplicate(df)
    print('remove message where the related category is flagged as 2.')
    df = df[df.related != 2]
    return df
    

def save_data(df, database_filename = 'figure_eight.db', table_name = 'msg_cat'):
    '''
    Usage: load the dataset into the database
        
    Args:
        df (dataframe): the dataframe you want to load
        database_filename (str): the database name
        table_name (str): the table name
            
    Returns:
        None
    '''
    print('Opening connection.')
    conn = sqlite3.connect(database_filename)
    print('Loading data to table "{}" in the database "{}".'.format(table_name, database_filename))
    df.to_sql(table_name, con = conn, if_exists='replace', index=False)
    # commit any changes to the database and close the database
    conn.commit()
    conn.close()
    print('Connection is closed.')
    return None

def read_db(db_name = 'figure_eight.db', tbl_name = 'msg_cat', limit = None):
    '''
    Usage: read data from the database
        
    Args:
        db_name (str): the database name
        tbl_name (str): the table name
        limit (int): the number of returned records, 
                     default - None (return all records)
            
    Returns:
        the dataframe from database queries
    '''
    print('Opening connection.')
    conn = sqlite3.connect(db_name)
    # get a cursor
    cur = conn.cursor()
    cmd = "SELECT * FROM " + tbl_name
    if limit is not None:
        cmd = cmd + ' limit ' + str(limit)
    print('Reading data from table "{}" in the database "{}".'.format(tbl_name, db_name))
    df = pd.read_sql(cmd, con = conn)
    conn.commit()
    conn.close()
    print('Connection is closed.')
    return df


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