# Trading Backtesting

## Overview
This project backtests on various trading strategies using stock data to select best trading strategy on **"GOOGL"** share and **investment amount of $10,000**. Types of trading strategies involved:
1. Basic Trading strategies (```src/basic.py```):
    - Simple Moving Average
    - Momentum Trading
    - Mean reversion
2. Machine learning  (```src/ml.py```):
    - Linear regression
    - Logistic regression
    - Neural networks

Best trading strategy is selected based on how much it outperforms the current market returns with highest. From there, a chart will be plotted to visualise how accurate the best trading strategy is compared to the market returns


## Configuration
1. Jupyter notebook is run inside VSC via a **virtual environment**:
    - Run ```python -m venv <NAME OF VENV>``` inside GitBash
    - Run command ```pip install jupyter notebook``` inside GitBash
2. Install the following required packages on Jupyter notebook inline:
    - ```%pip install pandas```
    - ```%pip install numpy```
    - ```%pip install quandl```
        - Create your own NASDAQ account [here](https://docs.data.nasdaq.com/v1.0/docs/python-installation) which allows you to have access to the python package ```Quandl``` 
        - Set your own API key inside python via anther file ```config.py```
    - ```%pip install matplotlib```
    - ```%pip install scikit-learn```
    - ```%pip install tensorflow```
    - ```%pip install keras```

    
