import pandas as pd


class DataProcessing:
    def create_raw_df(main_file_path):
        data = pd.read_csv(main_file_path)
        data.drop(['Adj Close', 'High','Low'], axis = 1, inplace = True)
        data.rename(columns={'Open':'open','Close': 'close', 'Date': 'date', 'Volume':'volume'}, inplace=True)
        data.index = pd.to_datetime(data['date'])
        data.drop(['date'], axis = 1, inplace = True)
        return data