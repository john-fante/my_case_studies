import pandas as pd
from river import compose, optim, metrics, preprocessing
from river.stream import iter_pandas
from river.linear_model import LinearRegression
from river.utils import Rolling


class OnlineLinearRegressionModel:
    def create_online_learning_pipeline(df:pd.DataFrame, l2_val:float, window_size:int=1, learning_rate:float=0.03, intercept_lr_val:float=0.1):
        print('the training process is starting')
        df['close_msft_shifted'] = df['close_msft'].shift(-1)
        df.drop(['close_msft'], axis=1, inplace=True)
        df.dropna(inplace=True)
        data = df.copy()
        # creating stream dataset
        # for prediction  the close price of the MSFT stock
        y = data.pop('close_msft_shifted')
        X_y_stream_dataset = iter_pandas(data, y)


        # --------------------  ONLINE/INCREMENTAL LEARNING MODEL  --------------------

        # creating Online/Incremental model with the River library
        # added columns for training
        selected_columns = set(list(data.columns))
        model = compose.Select(*selected_columns)

        # scaling (Actually, you need to use another scaler because there are negative values in the data, but it worked in my tests.)
        model |= preprocessing.MinMaxScaler()

        #final model
        model |= LinearRegression(l2=l2_val, intercept_lr = intercept_lr_val, optimizer=optim.Adam(learning_rate))


        # --------------------  TRAINING  --------------------
        # train metrics (rolling metrics for online learning)
        mae_metric = Rolling(metrics.MAE() , window_size = window_size)
        rmse_metric = Rolling(metrics.RMSE() , window_size = window_size)

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
        print('the training process is finished')
        print('the training results -> ' + str(mae_metric) + ' , ' + str(rmse_metric))
        return final_df, mae_metric, rmse_metric
