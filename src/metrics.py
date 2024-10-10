import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mad = mean_absolute_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true)).mean() * 100
    return mse, rmse, mad, mape
