import pandas as pd
import numpy as np

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