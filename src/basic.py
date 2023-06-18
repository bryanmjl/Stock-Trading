'''Helper Module for 3 basic trading strategies: SMA, Momentum, Mean reversion'''
import numpy as np

# Create a function for SMA of 2 periods
def SimpleMovingAverage(period1, period2, df, col, amount, tc):

    '''
    About
    =======================
    Function to generate buy/sell signals from Simple Moving Average Strategy
    Buy when shorter SMA is above longer SMA and sell when shorter SMA is below longer SMA

    Attributes
    ========================
    :period1: Size of rolling window of shorter SMA
    :period2: Size of rolling window of longer SMA
    :df: Which Dataset to be used
    :col: Which column to be used for analysis
    :amount: Base investment amount
    :tc: transaction costs

    Returns
    ========================
    Transformed dataframe df
    aperf as absolute performance of strategy itself
    operf as difference between strategy and base stock returns itself
    '''
    
    # Calculate SMA1 and SMA2 
    df['SMA1'] = df[col].rolling(period1).mean()
    df['SMA2'] = df[col].rolling(period2).mean()

    # Calculate Strategy returns
    df['position'] = np.where(df['SMA1'] > df['SMA2'], 1, -1)
    df = df.dropna()
    df['returns'] = np.log(df[col] / df[col].shift(1))
    df['strategy'] = df['position'].shift(1) * df['returns']

    # Determine when a trade take place and subtract transaction costs from the trade
    df.dropna(inplace= True)
    trades = df['position'].diff().fillna(0) != 0
    df['strategy'][trades] -= tc

    # Calculate cumulative returns
    df['creturns'] = amount * df['returns'].cumsum().apply(np.exp)
    df['cstrategy'] = amount * df['strategy'].cumsum().apply(np.exp)

    # Calculate absolute performance and whether we out/under perform strategy
    aperf = df['cstrategy'].iloc[-1]  
    operf = aperf - df['creturns'].iloc[-1]
    return df, aperf, operf


# Create Momentum Backtesting Strategy
def Momentum(n, df, col, amount, tc):
    
    '''
    About
    =======================
    Function to generate buy/sell signals from Momentum Strategy
    Buy when stock is last positive return + Sell when stock is last negative return

    Attributes
    ========================
    :n: Size of rolling window
    :df: Which dataset to be used
    :col: Which column to be used for analysis
    :amount: Base investment amount
    :tc: transaction costs

    Returns
    ========================
    Transformed dataframe df
    aperf as absolute performance of strategy itself
    operf as difference between strategy and base stock rteurns itself
    '''

    # Calculate returns
    df['returns'] = np.log(df[col] / df[col].shift(1))

    # Calculate buy sell signals and strategy returns
    df = df.dropna()
    df['position'] = np.sign(df['returns'].rolling(n).mean())  # Rolling window here
    df['strategy'] = df['position'].shift(1) * df['returns']

    # Determine when a trade take place and subtract transaction costs from the trade
    df.dropna(inplace= True)
    trades = df['position'].diff().fillna(0) != 0
    df['strategy'][trades] -= tc

    # Calculate cumulative returns
    df['creturns'] = amount * df['returns'].cumsum().apply(np.exp)
    df['cstrategy'] = amount * df['strategy'].cumsum().apply(np.exp)

    # Calculate absolute performance and whether we out/under perform strategy
    aperf = df['cstrategy'].iloc[-1]  
    operf = aperf - df['creturns'].iloc[-1]
    return df, aperf, operf


# Create MeanReversion Backtesting Strategy
def MeanReversion(n, df, col, threshold, amount, tc):

    '''
    About
    =======================
    Function to generate buy/sell signals from Mean Reversion Strategy
    Sell when stock return is above threshold and Buy and stock return is below threshold

    Attributes
    ========================
    :n: Size of rolling window
    :df: Which dataset to be used
    :col: Which column to be used for analysis
    :threshold: Threshold of stock return investment
    :amount: Base investment amount
    :tc: transaction costs

    Returns
    ========================
    Transformed dataframe df
    aperf as absolute performance of strategy itself
    operf as difference between strategy and base stock returns itself
    '''
    # Calculate returns SMA and threshold differences
    df['returns'] = np.log(df[col] / df[col].shift(1))
    df['SMA'] = df[col].rolling(n).mean() 
    df['distance'] = df[col] - df['SMA']

    # Calulate buy sell signals and strategy returns
    df.dropna(inplace = True)
    df['position'] = np.where(df['distance'] > threshold, -1, np.nan) # Short signal
    df['position'] = np.where(df['distance'] < -threshold, 1, df['position']) # Long signal
    df['position'] = np.where(df['distance'] * df['distance'].shift(1) < 0, 0, df['position']) # Neutral signal
    df['position'] = df['position'].ffill().fillna(0)
    df['strategy'] = df['position'].shift(1) * df['returns']

    # Determine when a trade take place and subtract transaction costs from the trade
    df.dropna(inplace= True)
    trades = df['position'].diff().fillna(0) != 0
    df['strategy'][trades] -= tc

    # Calculate cumulative returns
    df['creturns'] = amount * df['returns'].cumsum().apply(np.exp)
    df['cstrategy'] = amount * df['strategy'].cumsum().apply(np.exp)

    # Calculate absolute performance and whether we out/under perform strategy
    aperf = df['cstrategy'].iloc[-1]  
    operf = aperf - df['creturns'].iloc[-1]
    return df, aperf, operf


# Calculate drawdown period 
def drawdown(df, col):
    '''
    About
    =======================
    Function to calculate maximum drawdown period (How much an investment is down from peak before it recovers to peak)
    To be used in conjunction with functions such as "SimpleMovingAverage", "Momentum" or "MeanReversion", etc

    Attributes
    ========================
    :df: Which dataset to be used
    :col1: Cumulative strategy return aka "Cstrategy"

    Returns
    ========================
    Drawdown period -> Can be used as min or max, etc
    '''

    drawdown = df[col].cummax() - df[col]
    temp = drawdown[drawdown == 0]
    periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())  # Calculates differences between all dates
    return periods
