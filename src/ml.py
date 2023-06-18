'''Helper Module for ML strategies: OLS, Logistic Regression, Neural Networks'''
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score  # For logistic regression
from sklearn import linear_model

import tensorflow as tf              # For tensorflow models
from keras.models import Sequential  # Required model from Keras
from keras.layers import Dense       # Required layer from Keras
from keras.optimizers import Adam, RMSprop

# Ordinary Least Squares function
def OLS(df, target_col, lags, amount, tc):
    '''
    About
    =======================
    OLS Function (Ordinary Least Squares)
    Focus is on predicting direction of market movement 

    Attributes
    ========================
    :df: Which Dataset to be used
    :target_col: target column for calculating returns
    :lags: How many lags to be used
    :amount: Base investment amount
    :tc: transaction costs

    Returns
    ========================
    Transformed dataframe df with predicted values from OLS
    aperf as absolute performance of strategy itself
    operf as difference between strategy and base stock returns itself
    '''

    # Prepare dataset and calculate returns
    df['returns'] = np.log(df[target_col] / df[target_col].shift(1))
    df = df.dropna()

    # Prepare lags
    cols = []
    for lag in range(1, lags +1):
        col = f'lag_{lag}'
        df[col] = df['returns'].shift(lag)
        cols.append(col)

    df.dropna(inplace=True)

    # Prepare OLS model
    reg = np.linalg.lstsq(df[cols], 
                          np.sign(df['returns']),
                          rcond=None)[0]  # Get weights of variables based on lags

    # Backtest trading strategy
    # "Prediction" here refers to sign of hit ratio, which is sign of (product of market return and predicted return aka "cols")
    # For example, if sign of forecasted return is correct, it means the product is positive (+1 * +1 or -1*-1)
    df['prediction'] = np.sign(np.dot(df[cols], reg))  # df[cols] is stock return; reg is OLS prediction as a single 1xN array
    df['strategy'] = df['prediction'] * df['returns']

    # Determine when a trade occurs to subtact transaction costs each time a trade takes place
    trades = df['prediction'].diff().fillna(0) != 0
    df['strategy'][trades] -= tc

    # Subtract transaction costs from return when trade takes place
    df['creturns'] = amount * df['returns'].cumsum().apply(np.exp)
    df['cstrategy'] = amount * df['strategy'].cumsum().apply(np.exp)

    # Gross performance of strategy
    aperf = df['cstrategy'].iloc[-1]

    # Out or under-performance of strategy
    operf = aperf - df['creturns'].iloc[-1]

    return df, aperf, operf


# Logistic Regression
def logreg(df, target_col, lags, amount, tc):
    '''
    About
    =======================
    Logistic Regression as a classification model
    Focus is on predicting direction of market movement

    Attributes
    ========================
    :df: Which Dataset to be used
    :target_col: target column for calculating returns
    :lags: How many lags to be used
    :amount: Base investment amount
    :tc: transaction costs

    Returns
    ========================
    Transformed dataframe df with predicted values from logistic regression
    aperf as absolute performance of strategy itself
    operf as difference between strategy and base stock returns itself
    '''

    # Prepare dataset and calculate returns
    df['return'] = np.log(df[target_col] / df[target_col].shift(1))
    df.dropna(inplace=True)

    # Prepare lags
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag)  
        df[col] = df['return'].shift(lag)  
        cols.append(col) 
    df.dropna(inplace = True)

    # Prepare logistic regression model and implements fitting step
    lm = linear_model.LogisticRegression(C=1e7, solver='lbfgs',
                                              multi_class='auto',
                                              max_iter=1000)  # Initialise logistic regression model
    lm.fit(df[cols], np.sign(df['return']))  # Fit data (X,y) -> Sign of market movement as y variable
    
    # Backtest trading strategy
    df['prediction'] = lm.predict(df[cols])
    df['strategy'] = df['prediction'] * df['return']

    # Determine when a trade takes place and subtract transaction cost from return when it happens
    trades = df['prediction'].diff().fillna(0) != 0
    df['strategy'][trades] -= tc
    df['creturns'] = amount * df['return'].cumsum().apply(np.exp)
    df['cstrategy'] = amount * df['strategy'].cumsum().apply(np.exp)

    # Calculate absolute performance and out/underperformance of strategy
    aperf = df['cstrategy'].iloc[-1]
    operf = aperf - df['creturns'].iloc[-1]

    return df, aperf, operf


# Neural network
def neural_network(df, target_col, lags, amount, tc):
    '''
    About
    =======================
    Neural network model (Sequential) with layer (Dense)

    Attributes
    ========================
    :df: Which Dataset to be used
    :target_col: target column for calculating returns
    :lags: How many lags to be used
    :amount: Base investment amount
    :tc: transaction costs

    Returns
    ========================
    Transformed dataframe df with predicted values from neural network
    aperf as absolute performance of strategy itself
    operf as difference between strategy and base stock returns itself
    '''

    # Prepare dataset and calculate returns
    df['return'] = np.log(df[target_col] / df[target_col].shift(1))
    df['direction'] = np.where(df['return'] > 0, 1, 0)
    df = df.dropna()

    # Prepare lags
    cols = []
    for lag in range(1, lags +1):
        col = f'lag_{lag}'
        df[col] = df['return'].shift(lag)
        cols.append(col)
    
    # Initialise Keras Model (Adam) and Layers (Dense)
    optimizer = Adam(learning_rate=0.0001)
    model = Sequential()

    model.add(Dense(64, activation='relu',     # Add layers here; Relu is a type of activation function; 
                  input_shape=(lags,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Add layers here; Sigmoid is a type of activation function; 

    model.compile(optimizer=optimizer,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    
    # Compute training test data in 80-20 ratio
    training_length = int(len(df) * 0.8)
    training_data = df[:training_length].copy()
    mu, std = training_data.mean(), training_data.std()
    training_data_ = (training_data - mu) / std  # Normalise dataset under Gaussian distribution

    test_data = df[training_length:].copy()
    test_data_ = (test_data - mu) / std          # Normalise dataset under Gaussian distribution

    # Fit model with training_data
    model.fit(training_data[cols],
            training_data['direction'],
            epochs=50, verbose=False,
            validation_split=0.2, shuffle=False)
    
    # Get results
    res = pd.DataFrame(model.history.history)
    pred = np.where(model.predict(training_data_[cols]) > 0.5, 1, 0)  # Using normalised training data to predict
    training_data['prediction'] = np.where(pred > 0, 1, -1)  # Predict market direction back on in-sample training data (Non-normalised)
    training_data['strategy'] = (training_data['prediction'] * training_data['return'])

    # Determine when a trade occurs to subtact transaction costs each time a trade takes place
    trades = training_data['prediction'].diff().fillna(0) != 0
    training_data['strategy'][trades] -= tc

    # Subtract transaction costs from return when trade takes place
    training_data['creturns'] = amount * training_data['return'].cumsum().apply(np.exp)
    training_data['cstrategy'] = amount * training_data['strategy'].cumsum().apply(np.exp)

    # Calculate absolute performance and out/underperformance of strategy
    aperf = training_data['cstrategy'].iloc[-1]
    operf = aperf - training_data['creturns'].iloc[-1]

    return training_data, aperf, operf
