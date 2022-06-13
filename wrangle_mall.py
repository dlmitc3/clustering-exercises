import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#acquire
import acquire
from env import host, username, password
from pydataset import data

import sklearn.preprocessing

# Get Connection
def get_connection(db, username=username, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup SQL db.
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'
    

def get_sql_data(database,query):
    ''' 
        Take in a database and query
        check if csv exists for the queried database
        if it does read from the csv
        if it does not create the csv then read from the csv  
    '''
    
    if os.path.isfile(f'{database}_query.csv') == False:   # check for the file
        
        df = pd.read_sql(query, get_connection(database))  # create file 
        
        df.to_csv(f'{database}_query.csv',index = False)   # cache file
        
    return pd.read_csv(f'{database}_query.csv') # return contents of file


def get_mall_data():
    ''' acquire data from mall_customers database'''
    
    database = "mall_customers"

    query = "select * from customers"

    df = get_sql_data(database,query)
    
    return df

##################################Prepare##########################################

def detect_outliers(df, k, col_list, remove=False):
    ''' get upper and lower bound for list of columns in a dataframe 
        if desired return that dataframe with the outliers removed
    '''
    
    odf = pd.DataFrame()
    
    for col in col_list:

        q1, q2, q3 = df[f'{col}'].quantile([.25, .5, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        
        # print each col and upper and lower bound for each column
        print(f"{col}: Median = {q2} lower_bound = {lower_bound} upper_bound = {upper_bound}")

        # return dataframe of outliers
        odf = odf.append(df[(df[f'{col}'] < lower_bound) | (df[f'{col}'] > upper_bound)])
            
    return odf


def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[f'{col}'].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        return df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]  
    
    
def train_validate_test_split(df):
    '''split df into train, validate, test'''
    
    train, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    return train, validate, test


def min_max_scaling(train, validate, test, num_cols):
    '''
    Add scaled versions of a list of columns to train, validate, and test
    '''
    
    # reset index for merge 
    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    scaler = sklearn.preprocessing.MinMaxScaler() # create scaler object

    scaler.fit(train[num_cols]) # fit the object 

    # transform to get scaled columns
    train_scaled = pd.DataFrame(scaler.transform(train[num_cols]), columns = train[num_cols].columns + "_scaled")
    validate_scaled = pd.DataFrame(scaler.transform(validate[num_cols]), columns = validate[num_cols].columns + "_scaled")
    test_scaled = pd.DataFrame(scaler.transform(test[num_cols]), columns = test[num_cols].columns + "_scaled")
    
    # add scaled columns to dataframes
    train = train.merge(train_scaled, left_index=True, right_index=True)
    validate = validate.merge(validate_scaled, left_index=True, right_index=True)
    test = test.merge(train_scaled, left_index=True, right_index=True)
    
    return train, validate, test


def prepare_mall_data(df):
    ''' prepare mall data'''
    
    # split data
    train, validate, test = train_validate_test_split(df) 
       
    # encode gender in each column
    train = encoding(train, ['gender'], drop_first=True)
    validate = encoding(validate, ['gender'], drop_first=True)
    test = encoding(test, ['gender'], drop_first=True)
    
    # scale age, annual_income, and spending_score
    train, validate, test = min_max_scaling(train, validate, test,  ['age', 'annual_income', 'spending_score'])
    
    return train, validate, test