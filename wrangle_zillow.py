import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

def acquire():
    '''
    This function uses a SQL query to access the Codeup MySQL database and join 
    together all the relevant data from the zillow database.
    The data obtained includes all properties in the dataset which had a transaction in 2017.
    The function caches a csv in the local directory for later use. 
    '''
    # establish a filename for the local csv
    filename = 'zillow.csv'
    # check to see if a local copy already exists. 
    if os.path.exists(filename):
        print('Reading from local CSV...')
        # if so, return the local csv
        return pd.read_csv(filename)
    # otherwise, pull the data from the database:
    # establish database url
    url = env.get_db_url('zillow')
    # establish query
    sql = '''
            SELECT prop.*,
                   pred.logerror,
                   const.typeconstructiondesc,
                   arch.architecturalstyledesc,
                   land.propertylandusedesc,
                   heat.heatingorsystemdesc,
                   air.airconditioningdesc, 
                   bldg.buildingclassdesc,
                   story.storydesc
              FROM properties_2017 prop
                JOIN predictions_2017            pred  USING(parcelid)
                LEFT JOIN typeconstructiontype   const USING(typeconstructiontypeid)
                LEFT JOIN architecturalstyletype arch  USING(architecturalstyletypeid)
                LEFT JOIN propertylandusetype    land  USING(propertylandusetypeid)
                LEFT JOIN heatingorsystemtype    heat  USING(heatingorsystemtypeid)
                LEFT JOIN airconditioningtype    air   USING(airconditioningtypeid)
                LEFT JOIN buildingclasstype      bldg  USING(buildingclasstypeid)
                LEFT JOIN storytype              story USING(storytypeid)
              WHERE pred.transactiondate LIKE "2017%%"
                AND pred.transactiondate in (
                                             SELECT MAX(transactiondate)
                                               FROM predictions_2017
                                               GROUP BY parcelid
                                             )
                AND prop.latitude IS NOT NULL
                AND prop.longitude IS NOT NULL;
            '''
    print('No local file exists\nReading from SQL database...')
    # query the database and return the resulting table as a pandas dataframe
    df = pd.read_sql(sql, url)
    # save the dataframe to the local directory as a csv
    print('Saving to local CSV... ')
    df.to_csv(filename, index=False)
    # return the resulting dataframe
    return df

def prepare(df):

    id_cols = [col for col in df.columns if 'typeid' in col or col in ['id', 'parcelid']]
    df = df.drop(columns=id_cols)

    df = df[df.propertylandusedesc == 'Single Family Residential']

    columns_to_drop = ['calculatedbathnbr',
                   'finishedfloor1squarefeet',
                   'finishedsquarefeet12', 
                   'regionidcity',
                   'landtaxvaluedollarcnt',
                   'taxamount',
                   'rawcensustractandblock']
    df = df.drop(columns=columns_to_drop)

    columns_to_impute_zeros = ['fireplacecnt',
                           'garagecarcnt',
                           'garagetotalsqft',
                           'hashottuborspa',
                           'poolcnt',
                           'threequarterbathnbr',
                           'taxdelinquencyflag']

    for col in columns_to_impute_zeros:
        df[col] = np.where(df[col].isna(), 0, df[col])

    # add a feature: has_garage
    df['has_garage'] = np.where(df.garagecarcnt > 0, 1, 0)
    # add a feature: has_pool
    df['has_pool'] = np.where(df.poolcnt > 0, 1, 0)
    # add a feature: has_fireplace
    df['has_fireplace'] = np.where(df.fireplacecnt > 0, 1, 0)
    
    return df

        

def handle_missing_values(df, prop_required_column, prop_required_row):
    
    col_threshold = int(round(prop_required_column * len(df)))
    row_threshold = int(round(prop_required_row * len(df)))
    
    df = df.dropna(axis=1, thresh=row_threshold)
    #df = df.dropna(axis=0, thresh=col_threshold)
    
    return df

def get_row_nulls(df):    
    df2 = pd.DataFrame()
    df2['n_rows_null'] = df.isnull().sum()
    df2['pct_rows_null'] = df.isnull().mean()
    df2 = df2.reset_index()
    df2 = df2.rename(columns={'index': 'feature'})
    return df2

def get_column_nulls(df):   
    df2 = pd.DataFrame(df.isnull().sum(axis=1))
    df2.columns = ['n_cols_null']
    df2['pct_cols_null'] = df.isnull().mean(axis=1)
    return df2

def train_validate_test_split(df, test_size=.2, validate_size=.3, random_state=42):
    '''
    This function takes in a dataframe, then splits that dataframe into three separate samples
    called train, test, and validate, for use in machine learning modeling.
    Three dataframes are returned in the following order: train, test, validate. 
    
    The function also prints the size of each sample.
    '''
    # split the dataframe into train and test
    train, test = train_test_split(df, test_size=.2, random_state=42)
    # further split the train dataframe into train and validate
    train, validate = train_test_split(train, test_size=.3, random_state=42)
    # print the sample size of each resulting dataframe
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')

    return train, validate, test

def remove_outliers(train, validate, test, k, col_list):
    ''' 
    This function takes in a dataset split into three sample dataframes: train, validate and test.
    It calculates an outlier range based on a given value for k, using the interquartile range 
    from the train sample. It then applies that outlier range to each of the three samples, removing
    outliers from a given list of feature columns. The train, validate, and test dataframes 
    are returned, in that order. 
    '''
    # Create a column that will label our rows as containing an outlier value or not
    train['outlier'] = False
    validate['outlier'] = False
    test['outlier'] = False
    for col in col_list:

        q1, q3 = train[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        train['outlier'] = np.where(((train[col] < lower_bound) | (train[col] > upper_bound)) & (train.outlier == False), True, train.outlier)
        validate['outlier'] = np.where(((validate[col] < lower_bound) | (validate[col] > upper_bound)) & (validate.outlier == False), True, validate.outlier)
        test['outlier'] = np.where(((test[col] < lower_bound) | (test[col] > upper_bound)) & (test.outlier == False), True, test.outlier)

    # remove observations with the outlier label in each of the three samples
    train = train[train.outlier == False]
    train = train.drop(columns=['outlier'])

    validate = validate[validate.outlier == False]
    validate = validate.drop(columns=['outlier'])

    test = test[test.outlier == False]
    test = test.drop(columns=['outlier'])

    # print the remaining 
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')

    return train, validate, test