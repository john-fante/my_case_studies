import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from data_processing import DataProcessing
from feature_engineering import FeatureEngineering
from model import OnlineLinearRegressionModel
from model_tuning import HyperparameterOptimization


# data reading, processing
orcl_data = DataProcessing.create_raw_df('./case_study_stock_price/dataset/ORCL.csv')
ibm_data = DataProcessing.create_raw_df('./case_study_stock_price/dataset/IBM.csv')
microsoft_data = DataProcessing.create_raw_df('./case_study_stock_price/dataset/MSFT.csv')

merged_1 = pd.merge(orcl_data, ibm_data, left_index=True, right_index=True, how='outer')
merged_1.rename(columns={'close_x': 'close_orcl','open_x': 'open_orcl', 'volume_x': 'volume_orcl', 'close_y':'close_ibm', 'open_y': 'open_ibm','volume_y':'volume_ibm'}, inplace=True)

final_merged_df = pd.merge(merged_1, microsoft_data, left_index=True, right_index=True, how='outer')
final_merged_df.rename(columns={'close': 'close_msft', 'volume':'volume_msft','open': 'open_msft'}, inplace=True)

final_merged_df.dropna(inplace=True)




# feature engineering
feature_engineering = FeatureEngineering(df=final_merged_df)
df_added_time_series_features = feature_engineering.create_time_series_features()

df_added_rsi_feature = feature_engineering.create_rsi_features(column_name = 'close_orcl')
df_added_rsi_feature = feature_engineering.create_rsi_features(column_name = 'close_ibm')

df_added_macd_features = feature_engineering.create_macd_features(column_name = 'close_orcl')
df_added_macd_features = feature_engineering.create_macd_features(column_name = 'close_ibm')

df_added_features = df_added_macd_features.copy()
df_added_features.dropna(inplace=True)




# training, the best params with optuna
preds, mae, rmse = OnlineLinearRegressionModel.create_online_learning_pipeline(df_added_features,
                                                                               l2_val=0.14297374376783614,
                                                                               window_size=1,
                                                                               learning_rate=0.25435208069550486,
                                                                               intercept_lr_val=0.4975315164402574)


print('\n')
print('the last 10 prediction')
print(preds[-10:])



# hyperparamater tuning 
"""
optimization = HyperparameterOptimization(df=df_added_features)
best_params = optimization.optimize_model_parameters(n_trials=2500)
print(best_params)
"""

