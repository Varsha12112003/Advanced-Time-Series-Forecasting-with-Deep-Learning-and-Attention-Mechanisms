import numpy as np
import pandas as pd

def generate_nonlinear_nonstationary(length=2000, seed=0):
    np.random.seed(seed)
    t = np.arange(length)
    trend = 0.01 * (t**1.2)  # slow increasing trend (non-linear)
    seasonal = 5.0 * np.sin(2 * np.pi * t / 50)  # short seasonality
    hetero_noise = (1 + 0.005 * t) * np.random.randn(length)  # increasing noise
    series = 10 + trend + seasonal + hetero_noise
    return pd.Series(series, name='y')

if __name__ == '__main__':
    s = generate_nonlinear_nonstationary(1200)
    s.to_csv('synthetic_series.csv', index=False)
    print('Saved synthetic_series.csv')
