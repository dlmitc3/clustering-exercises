import pandas as pd
import numpy as np
import os
from env import host, username, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Acquire

# Get Connection
def get_connection(db, username=username, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup SQL db.
    '''
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

# Get Zillow Data
def zillow_data():
    '''
    This function reads the telco_churn data from the Codeup db into a df
    '''
    
    # Create SQL query
    sql_query = '''SELECT *                    
    FROM predictions_2017
    JOIN properties_2017 USING (parcelid)
    LEFT JOIN airconditioningtype USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    LEFT JOIN storytype USING (storytypeid)
    LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
    WHERE latitude IS NOT Null
        AND longitude IS NOT Null'''

    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))

    return df

# Cache'ing Zillow Data
def cache_zillow_data(cached=False):
    '''
    This function reads in Zillow data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in iris df from
    a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow_df.csv') == False:

        # Read fresh data from db into a DataFrame
        df = zillow_data()

        # Cache data
        df.to_csv('zillow_data.csv')

    else:

        # If csv file exists or cached == True, read in data from csv file.
        df = pd.read_csv('zillow_data.csv', index_col=0)
        
#### IRIS Data ####

def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT 
                    species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                FROM measurements
                JOIN species USING(species_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    
    return df

def get_iris_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('iris_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('iris_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_iris_data()
        
        # Cache data
        df.to_csv('iris_df.csv')
        
    return df

# Function that will return percent nulls in df by row

def null_df():
    # number of null values in df
    null = df.isnull().sum()
    # percent null by feature
    percent_null = df.isnull().sum()/df.shape[0]
    return pd.DataFrame({'null_rows': null, 'pct_null': percent_null}) # create/return dataframe

# PREPARE

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
        
        df = df[(df[f'{col}'] > lower_bound) & (df[f'{col}'] < upper_bound)]
        
    return df

#### Handle Missing Values

def handle_missing_values(df, col_perc, row_perc):
    ''' 
        take in a dataframe and percent null paramerters, returns df eliminating nulls by inputed threshold
    '''
    col_thresh = int(round(col_perc*df.shape[0],0)) # calc column threshold
    
    df.dropna(axis=1, thresh=col_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    row_thresh = int(round(row_perc*df.shape[1],0))  # calc row threshhold
    
    df.dropna(axis=0, thresh=row_thresh, inplace=True) # drop columns with non-nulls less than threshold
    
    return df

### split continuous data ###
def split_continuous(df):
    """
    Takes in a df
    Returns train, validate, and test DataFrames
    """
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # Take a look at your split datasets

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")
    return train, validate, test


def train_validate_test(df, target):
    """
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    """
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test

# Imputer

def impute(df, my_strategy, column_list):
    ''' take in a df, strategy, and cloumn list
        return df with listed columns imputed using input stratagy
    '''
        
    imputer = SimpleImputer(strategy=my_strategy)  # build imputer

    df[column_list] = imputer.fit_transform(df[column_list]) # fit/transform selected columns

    return df

### PREPARE ZILLOW DATA ###

def prepare_zillow(df):
    ''' Prepare Zillow Data'''
    
    # Restrict propertylandusedesc to those of single unit
    df = df[(df.propertylandusedesc == 'Single Family Residential') |
          (df.propertylandusedesc == 'Mobile Home') |
          (df.propertylandusedesc == 'Manufactured, Modular, Prefabricated Homes')]
    
    # remove outliers in bed count, bath count, and area to better target single unit properties
    df = remove_outliers(df, 1.5, ['calculatedfinishedsquarefeet', 'bedroomcnt', 'bathroomcnt'])
    
    # dropping cols/rows where more than half of the values are null
    df = handle_missing_values(df, prop_required_column = .5, prop_required_row = .5)
    
    # dropping the columns with 17K missing values too much to fill/impute/drop rows
    df = df.drop(columns=['heatingorsystemtypeid', 'buildingqualitytypeid', 'propertyzoningdesc', 'unitcnt', 'heatingorsystemdesc'])
    
    # Split data
    X_train, X_validate, X_test = acquire.split_continuous(df_dropna)
    
    # Organize Discrete & Continous features
     discrete_col = ['calculatedbathnbr', 'fullbathcnt', 'regionidcity', 'regionidzip', 'yearbuilt', 'censustractandblock']
cont_col = ['calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxamount']
    
    # imputing descreet columns with most frequent value
   
    X_train = impute(X_train, 'most_frequent', discrete_col)
    X_validate = impute(X_validate, 'most_frequent', discrete_col)
    X_test = impute(X_test, 'most_frequent', discrete_col)
    
    # imputing continuous columns with mean value
    X_train = impute(X_train, 'mean', cont_col)
    X_validate = impute(X_validate, 'mean', cont_col)
    X_test = impute(X_test, 'mean', cont_col)
    
    return X_train, X_validate, X_test