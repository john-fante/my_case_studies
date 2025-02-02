import pandas as pd
from river import compose, optim, metrics, preprocessing
from river.stream import iter_pandas
from river.linear_model import LinearRegression
from river.utils import Rolling
import optuna

class HyperparameterOptimization:
    def __init__(self, df):
        self.data=df.copy()
        self.data['close_msft_shifted'] = self.data['close_msft'].shift(-1)
        self.data.drop(['close_msft'], axis=1, inplace=True)
        self.data.dropna(inplace=True)

    # the same function for training
    def create_online_learning_pipeline_for_optuna(self, l2_val:float, window_size:int=1, learning_rate:float=0.03, intercept_lr_val:float=0.1):

        data = self.data
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
    

    def objective_func(self, trial):
        """
        Objective function for hyperparameter optimization using Optuna.

        This function defines the hyperparameters that Optuna will optimize and evaluates the model's performance
        with different combinations of these hyperparameters. It computes the average of two error metrics (RMSE and MAE)
        to guide the optimization process.

        The function works as follows:
        1. Optuna suggests values for the following hyperparameters:
            - `window_size`: The size of the moving window used for training (integer between 1 and 14).
            - `learning_rate`: The rate at which the model learns (float between 0.01 and 0.5, on a logarithmic scale).
            - `intercept_lr_val`: Learning rate for the model intercept (float between 0.1 and 1.2, on a logarithmic scale).
            - `l2_val`: The L2 regularization strength (float between 0.1 and 1.5, on a logarithmic scale).

        2. The model is trained using the suggested hyperparameters, and two evaluation metrics (RMSE and MAE) are calculated.

        3. The function returns the average of RMSE and MAE, which Optuna uses to determine how well the model performs 
        with the given hyperparameters. The lower the average, the better the hyperparameters.

        Args:
            trial (optuna.trial.Trial): The trial object from Optuna that handles suggesting hyperparameter values.

        Returns:
            float: The average of RMSE and MAE error metrics, which Optuna tries to minimize.
        """
        window_size = trial.suggest_int('window_size', 1, 14)
        learning_rate = trial.suggest_float('learning_rate', 1e-2, 0.5, log=True)
        intercept_lr_val = trial.suggest_float('intercept_lr_val', 1e-1, 1.2, log=True)
        l2_val = trial.suggest_float('l2_val', 1e-1, 1.5, log=True)

        rmse_metric, mae_metric = self.create_online_learning_pipeline_for_optuna(self.data, l2_val=l2_val, window_size=window_size,learning_rate=learning_rate,intercept_lr_val=intercept_lr_val)
        return (mae_metric+rmse_metric)/2


    def optimize_model_parameters(self, n_trials:int = 2500):
        """
        Optimize the model's hyperparameters using Optuna.

        This function performs a hyperparameter optimization process using Optuna to find the best combination of
        hyperparameters for the model. The optimization process minimizes the average of RMSE and MAE error metrics
        by calling the `objective_func` method, which evaluates the model’s performance based on different hyperparameter
        configurations.

        The process involves the following steps:
        1. **Study Creation**: A new Optuna study is created with the goal of minimizing the objective function.
        2. **Optimization**: Optuna performs `n_trials` number of trials, each trial suggesting a set of hyperparameters and
        evaluating the model's performance using `objective_func`.
        3. **Best Parameters**: After completing the trials, the function returns the hyperparameter configuration that resulted in
        the best model performance (i.e., the lowest average of RMSE and MAE).

        Args:
            n_trials (int, optional): The number of optimization trials to run. Default is 2500.

        Returns:
            dict: A dictionary containing the best hyperparameters found by Optuna. The keys correspond to the hyperparameter names,
                and the values are the optimized values.
        """
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective_func, n_trials=n_trials)
        return study.best_params
