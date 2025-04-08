# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:41:52 2025

@author: Ronyt
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

# Example: Seasonal data (10 periods)
time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
data = np.array([10, 15, 12, 18, 14, 11, 16, 13, 19, 15])  # Example seasonal pattern

df = pd.DataFrame({'Time': time, 'Value': data})

# Decompose the time series
result = seasonal_decompose(df['Value'], model='additive', period=5, extrapolate_trend='freq')
trend = result.trend.dropna()
seasonal = result.seasonal
deseasonalized = df['Value'] - seasonal

# Fit linear regression on trend
valid_indices = trend.dropna().index.values.reshape(-1, 1)
trend_values = trend.dropna().values.reshape(-1, 1)
model = LinearRegression()
model.fit(valid_indices, trend_values)

# Predict next period
next_time = np.array([df['Time'].max() + i for i in range(1, 6)]).reshape(-1, 1)
trend_forecast = model.predict(next_time)
seasonal_forecast = seasonal[:5].values  # Repeating seasonal pattern
forecast = trend_forecast + seasonal_forecast

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(df['Time'], df['Value'], label='Original Data', marker='o')
plt.plot(df['Time'], trend, label='Trend', linestyle='dashed')
plt.scatter(next_time, forecast, color='red', label='Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Seasonal Forecast using Decomposition')
plt.show()
