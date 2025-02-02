# -*- coding: utf-8 -*-
"""case-study-dataguess-ekin-bozyel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1g8AAHZ-EEkT03IQl4HR_iZLlM_-T0vc0

# Microsoft Stock Price Prediction with Online/Incremental Learning


Ekin Bozyel
<br>

* Phone: 0534 328 27 56
* Mail address : ekinbozyel@gmail.com
* Linkedin account: (www.linkedin.com/in/ekin-bozyel-453934269)
* Github account : (https://github.com/john-fante)
* Kaggle account : (https://www.kaggle.com/banddaniel)
* Stack Overflow account :  (https://stackoverflow.com/users/22880135)

<hr>

I developed a solution using the steps below

* Data finding (dataset -> https://www.kaggle.com/datasets/benjaminpo/s-and-p-500-with-dividends-and-splits-daily-updated), data reading, preprocessing
* In addition to the Microsoft stock, I used the same domain 2 stock prices (Oracle and IBM) (I only used daily closing price and volume data), first I added Time Series Features(weekly, daily, quarterly), then I added RSI and MACD Features for other shares except Microsoft.
* I used tuned LinearRegression model from river library(online learning package).
* Evaluation metrics -> MAE and RMSE (the result MAE: 0.00301 , RMSE: 0.00301) (By looking at both MAE and RMSE together, you can get a better idea of your model's performance in terms of both average error (MAE) and the severity of large errors (RMSE).)
* Hyperparameter tuned with optuna.

## References
* https://en.wikipedia.org/wiki/Online_machine_learning
* https://riverml.xyz/dev/api/overview/
* https://optuna.org
"""

# Installing river package for online learning
!pip install river -q

# Importing dependencies

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import plotly.express as px

from river import compose, optim, metrics, preprocessing
from river.stream import iter_pandas
from river.linear_model import LinearRegression
from river.utils import Rolling

import optuna

"""# 1) Reading and Preparing Raw Data"""

# Creating raw df with dropping some features, I only use Close, Date and Volume features

def create_raw_df(main_file_path):
    data = pd.read_csv(main_file_path)
    data.drop(['Adj Close', 'High','Low'], axis = 1, inplace = True)
    data.rename(columns={'Open':'open','Close': 'close', 'Date': 'date', 'Volume':'volume'}, inplace=True)
    data.index = pd.to_datetime(data['date'])
    data.drop(['date'], axis = 1, inplace = True)
    return data

# orcl_data -> Oracle stock history up to date today
# ibm_data -> IBM stock history up to date today
# microsoft_data -> Microsoft stock history up to date today

orcl_data = create_raw_df('/kaggle/input/s-and-p-500-with-dividends-and-splits-daily-updated/ORCL.csv')
ibm_data = create_raw_df('/kaggle/input/s-and-p-500-with-dividends-and-splits-daily-updated/IBM.csv')
microsoft_data = create_raw_df('/kaggle/input/s-and-p-500-with-dividends-and-splits-daily-updated/MSFT.csv')

merged_1 = pd.merge(orcl_data, ibm_data, left_index=True, right_index=True, how='outer')
merged_1.rename(columns={'close_x': 'close_orcl','open_x': 'open_orcl', 'volume_x': 'volume_orcl', 'close_y':'close_ibm', 'open_y': 'open_ibm','volume_y':'volume_ibm'}, inplace=True)

final_merged_df = pd.merge(merged_1, microsoft_data, left_index=True, right_index=True, how='outer')
final_merged_df.rename(columns={'close': 'close_msft', 'volume':'volume_msft','open': 'open_msft'}, inplace=True)

# drop na values
final_merged_df.dropna(inplace=True)

# final data (IBM + Oracle + Microsoft, close prices and volumes)
final_merged_df.tail()

"""# 2) Feature Engineering

"""

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()


    # a function for creating time series features
    def create_time_series_features(self) -> pd.DataFrame:
        """
        Generates new time-related features from a DataFrame with a datetime index.

        The function extracts the following information from the datetime index:
        - Day of the week (week number)
        - Month of the year
        - Quarter of the year

        Parameters:
        ----------
        df : pandas.DataFrame
            A DataFrame with a datetime index. The index is expected to represent time
            (e.g., `pd.to_datetime()`).

        Returns:
        -------
        pandas.DataFrame
            A DataFrame with the original data and new cyclic features:
            - `week_sin`: Sine transformation of the week (day of the week).
            - `week_cos`: Cosine transformation of the week (day of the week).
            - `month_sin`: Sine transformation of the month.
            - `month_cos`: Cosine transformation of the month.
            - `quarter_sin`: Sine transformation of the quarter.
            - `quarter_cos`: Cosine transformation of the quarter.
        """

        df = self.df
        df['week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        # creating cycles features
        df['week_sin'] = np.sin(df['week']*(2.*np.pi/7))
        df['week_cos'] = np.sin(df['week']*(2.*np.pi/7))

        df['month_sin'] = np.sin(df['month']*(2.*np.pi/12))
        df['month_cos'] = np.sin(df['month']*(2.*np.pi/12))

        df['quarter_sin'] = np.sin(df['quarter']*(2.*np.pi/4))
        df['quarter_cos'] = np.sin(df['quarter']*(2.*np.pi/4))

        df.drop(['week', 'month' ,'quarter'], axis = 1, inplace = True)

        return df


    # a function for calculating rsi with default args
    def create_rsi_features(self, column_name:str, window:int=14) -> pd.DataFrame:
        """
        Calculate the Relative Strength Index (RSI) for a given column in a DataFrame.
        The function computes the RSI, a momentum oscillator that measures the speed and
        change of price movements. The RSI values are added as a new column in the input
        DataFrame with the name `<column_name>_rsi`.

        Parameters:
        ----------
        df : pd.DataFrame
            A pandas DataFrame containing time series data. The DataFrame should have at least
            one numeric column with the specified `column_name` for which RSI will be calculated.

        column_name : str
            The name of the column in the DataFrame for which the RSI will be calculated.

        window : int, default=14
            The number of periods (days) used for calculating the rolling average of gains and losses.
            The default value is 14, which is the typical period used in most RSI calculations.

        Returns:
        -------
        pd.DataFrame
            The original DataFrame with an additional column named `<column_name>_rsi`,
            which contains the calculated RSI values.

        """
        df = self.df
        delta = df[column_name].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))
        df[str(column_name)+'_rsi'] = rsi
        return df

    # a function for calculating macd with default args
    def create_macd_features(self, column_name, short_window:int=12, long_window:int=26, signal_window:int=9) -> pd.DataFrame:
        """
        Calculate the Moving Average Convergence Divergence (MACD) and related features for a given column in a DataFrame.

        The function computes the MACD, Signal Line, and Histogram for the specified column in the input DataFrame.
        The MACD is calculated by subtracting the long-term exponential moving average (EMA) from the short-term EMA.
        The Signal Line is the EMA of the MACD. The Histogram represents the difference between the MACD and the Signal Line.

        Parameters:
        ----------
        df : pd.DataFrame
            A pandas DataFrame containing time series data. The DataFrame should have at least
            one numeric column with the specified `column_name` for which MACD will be calculated.

        column_name : str
            The name of the column in the DataFrame for which the MACD, Signal Line, and Histogram will be calculated.

        short_window : int, default=12
            The window (in days) used for calculating the short-term exponential moving average (EMA).
            The default value is 12, which is commonly used in most MACD calculations.

        long_window : int, default=26
            The window (in days) used for calculating the long-term exponential moving average (EMA).
            The default value is 26, which is typically used in MACD calculations.

        signal_window : int, default=9
            The window (in days) used for calculating the Signal Line, which is the EMA of the MACD.
            The default value is 9, which is a standard period for the Signal Line.

        Returns:
        -------
        pd.DataFrame
            The original DataFrame with the following new columns:
            - `<column_name>_macd`: The MACD value, which is the difference between the short-term and long-term EMAs.
            - `<column_name>_signal_line`: The Signal Line, which is the EMA of the MACD.
            - `<column_name>_histogram`: The Histogram, representing the difference between the MACD and the Signal Line.
        """
        df = self.df
        ema_short = df[column_name].ewm(span=short_window, adjust=False).mean()
        ema_long = df[column_name].ewm(span=long_window, adjust=False).mean()
        macd = ema_short - ema_long

        signal_line = macd.ewm(span=signal_window, adjust=False).mean()

        histogram = macd - signal_line

        df[str(column_name)+'_macd'] = macd
        df[str(column_name)+'_signal_line'] = signal_line
        df[str(column_name)+'_histogram'] = histogram
        return df

# adding feature engineering methods
feature_engineering = FeatureEngineering(df=final_merged_df)
df_added_time_series_features = feature_engineering.create_time_series_features()

df_added_rsi_feature = feature_engineering.create_rsi_features(column_name = 'close_orcl')
df_added_rsi_feature = feature_engineering.create_rsi_features(column_name = 'close_ibm')

df_added_macd_features = feature_engineering.create_macd_features(column_name = 'close_orcl')
df_added_macd_features = feature_engineering.create_macd_features(column_name = 'close_ibm')

df_added_features = df_added_macd_features.copy()

# drop nan variables
df_added_features.dropna(inplace=True)
df_added_features.tail()

# final dateframe for training
df_added_features.describe()

# final training data info with columns
df_added_features.info()

"""# 3) LinearRegression Model and Online/Incremental Training Pipeline"""

def create_online_learning_pipeline(df:pd.DataFrame, l2_val:float, window_size:int=1, learning_rate:float=0.03, intercept_lr_val:float=0.1):

    df['close_msft_shifted'] = df['close_msft'].shift(-1)
    df.drop(['close_msft'], axis=1, inplace=True)
    df.dropna(inplace=True)
    data = df.copy()
    print(data.shape)
    # creating stream dataset
    # for prediction  the close price of the MSFT stock
    y = data.pop('close_msft_shifted')
    X_y_stream_dataset = iter_pandas(data, y)


    # --------------------  ONLINE/INCREMENTAL LEARNING MODEL  --------------------

    # creating Online/Incremental model with the River library
    # added columns for training
    selected_columns = set(list(data.columns))
    print("columns for training")
    print(selected_columns)
    print('\n')
    model = compose.Select(*selected_columns)

    # scaling (Actually, you need to use another scaler because there are negative values in the data, but it worked in my tests.)
    model |= preprocessing.MinMaxScaler()

    #final model
    model |= LinearRegression(l2=l2_val, intercept_lr = intercept_lr_val, optimizer=optim.Adam(learning_rate))


    # --------------------  TRAINING  --------------------
    # train metrics (rolling metrics for online learning)
    mae_metric = Rolling(metrics.MAE() , window_size = window_size)
    rmse_metric = Rolling(metrics.RMSE() , window_size = window_size)

    dates = data.index
    y_trues = []
    y_preds = []

    # training loop
    for x, y in X_y_stream_dataset:

        y_pred = model.predict_one(x)
        if y_pred < 0: y_pred = 0  # for minus value validation
        #learn only one sample
        model.learn_one(x, y)
        mae_metric.update(y, y_pred)
        rmse_metric.update(y, y_pred)

        y_trues.append(y)
        y_preds.append(y_pred)

    # final dataframe with predictions
    final_df = pd.DataFrame({'time': data.index,'true': y_trues, 'prediction': y_preds}, index = data.index)
    print(str(mae_metric) + ' , ' + str(rmse_metric))
    return final_df, mae_metric, rmse_metric

"""# 4) Prediction Plotting (day by day prediction)"""

# a function for plotting predictions and ground truths
# data -> data frame
# plot_title -> title of the graph
def plot_predictions(data, plot_title):
    fig = px.line(data, x='time', y=['prediction','true'])

    fig.update_layout(xaxis_range=['2024-01-01','2025-01-30'], title_text= plot_title +" - after January 2024")

    fig.update_xaxes(rangeslider_visible=True)
    fig.show()

"""# 5) Training"""

# training, the best params with optuna
preds, mae, rmse = create_online_learning_pipeline(df_added_features,
                                                   l2_val=0.14297374376783614,
                                                   window_size=1,
                                                   learning_rate=0.25435208069550486,
                                                   intercept_lr_val=0.4975315164402574)

"""# 6) Prediction and Results"""

# predictions plot
plot_predictions(preds, 'the window size is 1 (daily)')

# random 10 predictions
# each prediction is estimated according to the data up to itself
preds.sample(15, random_state = 999).sort_values(by='date')

# the last 10 day predictions
# each prediction is estimated according to the data up to itself

preds[-10:]

"""# 6) BONUS: Hyperparameter Optimization with Optuna"""

def create_online_learning_pipeline_for_optuna(df:pd.DataFrame, l2_val:float, window_size:int=1, learning_rate:float=0.03, intercept_lr_val:float=0.1):

    data = df.copy()
    # creating stream dataset
    # for prediction  the close price of the MSFT stock
    y = data.pop('close_msft_shifted')
    X_y_stream_dataset = iter_pandas(data, y)


    # --------------------  ONLINE/INCREMENTAL LEARNING MODEL  --------------------

    # creating Online/Incremental model with the River library
    # added columns for training
    model = compose.Select(*set(list(data.columns)))

    # scaling (Actually, you need to use another scaler because there are negative values in the data, but it worked in my tests.)
    model |= preprocessing.MinMaxScaler()

    #final model
    model |= LinearRegression(l2=l2_val, intercept_lr = intercept_lr_val, optimizer=optim.Adam(learning_rate))



    # --------------------  TRAINING  --------------------
    # train metrics (rolling metrics for online learning)
    mae_metric = Rolling(metrics.MAE() , window_size = window_size)
    rmse_metric = Rolling(metrics.RMSE() , window_size = window_size)

    dates = data.index
    y_trues = []
    y_preds = []

    # training loop
    for x, y in X_y_stream_dataset:

        y_pred = model.predict_one(x)
        if y_pred < 0: y_pred = 0  # for minus value validation
        #learn only one sample
        model.learn_one(x, y)
        mae_metric.update(y, y_pred)
        rmse_metric.update(y, y_pred)

        y_trues.append(y)
        y_preds.append(y_pred)


    return mae_metric.get(), rmse_metric.get()

# objective function
# I tried to minimize the average of MAE and RMSE values to minimize the lost function. -> MINIMIZE [(mae_metric+rmse_metric)/2]


df_for_tuning = df_added_features.copy()
df_for_tuning['close_msft_shifted'] = df_for_tuning['close_msft'].shift(-1)
df_for_tuning.drop(['close_msft'], axis=1, inplace=True)
df_for_tuning.dropna(inplace=True)

def objective_func(trial):
    window_size = trial.suggest_int('window_size', 1, 14)
    learning_rate = trial.suggest_float('learning_rate', 1e-2, 0.5, log=True)
    intercept_lr_val = trial.suggest_float('intercept_lr_val', 1e-1, 1.2, log=True)
    l2_val = trial.suggest_float('l2_val', 1e-1, 1.5, log=True)

    rmse_metric, mae_metric = create_online_learning_pipeline_for_optuna(df_for_tuning,l2_val=l2_val, window_size=window_size,learning_rate=learning_rate,intercept_lr_val=intercept_lr_val)
    return (mae_metric+rmse_metric)/2

study = optuna.create_study(direction='minimize')

# trials
study.optimize(objective_func, n_trials=2500)

# the best params
study.best_params #0.0007130734483666856