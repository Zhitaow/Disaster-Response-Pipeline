import pandas as pd
import numpy as np
import sqlite3

class ETL_pipeline():
    # load dataset
    def extract(self, msg_file = 'messages.csv', cat_file = 'categories.csv'):
        '''
        Usage: read messages and categories csv files into two dataframe objects
        
        Args:
            msg_file (str): filepath and filename for messages file 
            cat_file (str): filepath and filename for categories file 
    
        Returns:
            two dataframes: df_messages, df_categories
    
        '''
        df_messages = pd.read_csv(msg_file, delimiter = ',')
        df_categories = pd.read_csv(cat_file, delimiter = ',')
        return df_messages, df_categories
    
    def deduplicate(self, df):
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
    
    def transform(self, df_messages, df_categories):
        '''
        Usage: merge two datasets of messages categories, and transform into one dataframe with customized format
        
        Args:
            df_messages (dataframe): the dataframe of message dataset
            df_categories (dataframe): the dataframe of category dataset
            
        Returns:
            a cleaned dataset (dataframe)
        '''
        print('Merging two datasets.')
        # merge datasets
        df_merge = df_messages.merge(df_categories, how = 'outer', on = 'id')
        # create a dataframe of the 36 individual category columns
        categories = pd.DataFrame(data = np.zeros((df_merge.shape[0], 36), dtype = int))
        # split categories into separate category columns
        category_colnames = df_merge.iloc[0] \
                            .str.split(';', expand = True).loc['categories'].apply(lambda x : x.split('-')[0])
        # rename the columns of `categories`
        categories.columns = category_colnames
        print('Parsing category values row-wise.')
        # convert category values row-wise
        for irow, row in enumerate(df_merge.iterrows()):
            # list of categories in each row of df.categories
            cat_list = row[1][4].split(';')
            # only keep the last letter, which is "1" or "0"
            cat_list = [x[-1] for x in cat_list]
            # convert from list to integer array
            cat_array = np.array(cat_list).astype(int)
            # set array values to categories dataframe by row index
            categories.iloc[irow,:] = cat_array
        # drop the original categories column from `df`
        df_merge.drop(labels = 'categories', axis = 1, inplace = True)
        # concatenate the original dataframe with the new `categories` dataframe
        df_merge = pd.concat([df_merge, categories], axis = 1)
        print('Removing duplicated values.')
        # duplication
        self.deduplicate(df_merge)
        return df_merge
    
    def load_db(self, df_results, db_name = 'figure_eight.db', tbl_name = 'msg_cat'):
        '''
        Usage: load the dataset into the database
        
        Args:
            df_results (dataframe): the dataframe you want to load
            db_name (str): the database name
            tbl_name (str): the table name
            
        Returns:
            None
        '''
        print('Opening connection.')
        conn = sqlite3.connect(db_name)
        print('Loading data to table "{}" in the database "{}".'.format(tbl_name, db_name))
        df_results.to_sql(tbl_name, con = conn, if_exists='replace', index=False)
        # commit any changes to the database and close the database
        conn.commit()
        conn.close()
        print('Connection is closed.')
        return None
    
    def read_db(self, db_name = 'figure_eight.db', tbl_name = 'msg_cat', limit = None):
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