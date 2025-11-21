# Advanced Time Series Forecasting with Deep Learning and Attention - Project Package

Contents:
- README.md (this file)
- requirements.txt - Python dependencies
- data_generation.py - scripts to generate synthetic non-stationary noisy univariate series and example dataset loaders
- model.py - PyTorch implementation of a Seq2Seq LSTM encoder-decoder with Luong attention and a Transformer encoder example
- train.py - training loop with hyperparameter options and evaluation metrics (RMSE, MAE, MAPE)
- evaluate.py - evaluation script to compare model against ARIMA baseline (uses statsmodels)
- example_usage.sh - short shell commands to run data generation, training, and evaluation

Notes:
- This package is a starting point. It focuses on clarity and reproducibility.
- Ensure you run in a Python environment with the listed requirements.
- The model implementation and scripts are intentionally simple and well-commented for study and extension.

