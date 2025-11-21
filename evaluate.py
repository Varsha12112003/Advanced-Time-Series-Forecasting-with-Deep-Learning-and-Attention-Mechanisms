import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from statsmodels.tsa.arima.model import ARIMA
from data_generation import generate_nonlinear_nonstationary

def rmse(a,b): return math.sqrt(mean_squared_error(a,b))
def mape(a,b): return np.mean(np.abs((a-b)/a)) * 100

def evaluate_arima(series, train_size=0.8, order=(5,1,0), forecast_horizon=12):
    n = len(series)
    split = int(train_size * n)
    train = series[:split]
    test = series[split:split+forecast_horizon]
    model = ARIMA(train, order=order).fit()
    preds = model.forecast(steps=forecast_horizon)
    return {'RMSE': rmse(test, preds), 'MAE': mean_absolute_error(test, preds), 'MAPE': mape(test, preds)}

if __name__ == '__main__':
    s = generate_nonlinear_nonstationary(600)
    res = evaluate_arima(s, forecast_horizon=12)
    print('ARIMA baseline:', res)
