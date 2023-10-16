#import the needed librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
import pandas as pd
"""
This fonction serves to clean the data for trainining and it will split it into train and test needed for further steps
"""


def transform_data(df):
    X = df.loc[(df["type"] == 'TRANSFER') | (df["type"]== 'CASH_OUT')]

    randomState = 5
    np.random.seed(randomState)

#X = X.loc[np.random.choice(X.index, 100000, replace = False)]

    Y = X['isFraud']
    del X['isFraud']

# Eliminate columns shown to be irrelevant for analysis in the EDA
    X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

# Binary-encoding of labelled data in 'type'
    X.loc[X['type'] == 'TRANSFER', 'type'] = 0
    X.loc[X['type'] == 'CASH_OUT', 'type'] = 1
    X.type = X.type.astype(int) # convert dtype('O') to dtype(int)
    Xfraud = X.loc[Y == 1]
    XnonFraud = X.loc[Y == 0]
    X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0), \
      ['oldBalanceDest', 'newBalanceDest']] = - 1
    X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0), \
      ['oldBalanceOrig', 'newBalanceOrig']] = np.nan
    # feature engineering 
    X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
    X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest
    
    X = df.drop('isFraud', axis=1)  # Select features
    y = df["isFraud"] # Target
     
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)  # Split data 80/20
    #scaler = MinMaxScaler()  # Normalize train & test features
    #X_train[num_features] = scaler.fit_transform(X_train[num_features])
    #X_test[num_features] = scaler.transform(X_test[num_features])
    return X_train,X_test,y_train,y_test