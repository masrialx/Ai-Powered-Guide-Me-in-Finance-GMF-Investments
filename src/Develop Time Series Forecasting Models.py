# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings for cleaner output

# Set plot style
plt.style.use('seaborn')

# Task 2: Develop Time Series Forecasting Models
# Step 1: Load cleaned data from Task 1
data = pd.read_csv('cleaned_adj_close.csv', index_col='Date', parse_dates=True)
tsla_data = data['TSLA'].dropna()  # Focus on TSLA for forecasting

# Step 2: Divide dataset into training and testing sets
train_size = int(len(tsla_data) * 0.8)  # 80% train, 20% test
train_data, test_data = tsla_data[:train_size], tsla_data[train_size:]

print(f"Training data size: {len(train_data)}")
print(f"Testing data size: {len(test_data)}")

# Step 3: Train an ARIMA model using auto_arima for parameter optimization
# Use pmdarima's auto_arima to find the best (p, d, q) parameters
auto_model = pm.auto_arima(train_data, 
                          start_p=1, start_q=1, 
                          max_p=5, max_q=5, 
                          d=None,  # Let auto_arima determine differencing
                          seasonal=False,  # ARIMA (no seasonality); set True for SARIMA
                          trace=True,  # Show parameter search progress
                          error_action='ignore', 
                          suppress_warnings=True, 
                          stepwise=True)

# Print the best model's parameters
print("\n=== Best ARIMA Model Summary ===")
print(auto_model.summary())

# Step 4: Fit the ARIMA model with optimized parameters
p, d, q = auto_model.order
arima_model = ARIMA(train_data, order=(p, d, q))
arima_fit = arima_model.fit()

# Step 5: Forecast on the test set
forecast = arima_fit.forecast(steps=len(test_data))

# Convert forecast to a pandas Series for easier handling
forecast_index = test_data.index
forecast_series = pd.Series(forecast, index=forecast_index)

# Step 6: Evaluate model performance
mae = mean_absolute_error(test_data, forecast_series)
rmse = np.sqrt(mean_squared_error(test_data, forecast_series))
mape = np.mean(np.abs((test_data - forecast_series) / test_data)) * 100

print("\n=== Model Evaluation Metrics ===")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Step 7: Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Test Data', color='orange')
plt.plot(forecast_series, label='ARIMA Forecast', color='green')
plt.title('TSLA Stock Price Forecast (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.show()

# Optional: Extend to SARIMA (uncomment to use)
"""
# SARIMA Model with seasonality (e.g., yearly: 252 trading days)
sarima_model = pm.auto_arima(train_data, 
                             start_p=1, start_q=1, 
                             max_p=3, max_q=3, 
                             d=None, 
                             seasonal=True, 
                             m=252,  # Yearly seasonality
                             start_P=0, max_P=2, 
                             start_Q=0, max_Q=2, 
                             trace=True, 
                             error_action='ignore', 
                             suppress_warnings=True, 
                             stepwise=True)

sarima_fit = ARIMA(train_data, order=sarima_model.order, seasonal_order=sarima_model.seasonal_order).fit()
sarima_forecast = sarima_fit.forecast(steps=len(test_data))
sarima_forecast_series = pd.Series(sarima_forecast, index=test_data.index)

# Evaluate SARIMA
sarima_mae = mean_absolute_error(test_data, sarima_forecast_series)
sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_forecast_series))
sarima_mape = np.mean(np.abs((test_data - sarima_forecast_series) / test_data)) * 100
print("\n=== SARIMA Evaluation Metrics ===")
print(f"MAE: {sarima_mae:.2f}, RMSE: {sarima_rmse:.2f}, MAPE: {sarima_mape:.2f}%")
"""

# Optional: LSTM Model (uncomment to use; requires tensorflow)
"""
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Prepare data for LSTM
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))
scaled_test = scaler.transform(test_data.values.reshape(-1, 1))

# Create sequences for LSTM (e.g., 60-day lookback)
lookback = 60
X_train, y_train = [], []
for i in range(lookback, len(scaled_train)):
    X_train.append(scaled_train[i-lookback:i, 0])
    y_train.append(scaled_train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Prepare test data for prediction
X_test = []
for i in range(lookback, len(scaled_test)):
    X_test.append(scaled_test[i-lookback:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Forecast with LSTM
lstm_pred = lstm_model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred)
lstm_forecast_series = pd.Series(lstm_pred.flatten(), index=test_data.index[lookback:])

# Evaluate LSTM
lstm_mae = mean_absolute_error(test_data[lookback:], lstm_forecast_series)
lstm_rmse = np.sqrt(mean_squared_error(test_data[lookback:], lstm_forecast_series))
lstm_mape = np.mean(np.abs((test_data[lookback:] - lstm_forecast_series) / test_data[lookback:])) * 100
print("\n=== LSTM Evaluation Metrics ===")
print(f"MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}, MAPE: {lstm_mape:.2f}%")
"""

# Save forecast results for Task 3
forecast_series.to_csv('tsla_arima_forecast.csv')
print("\nARIMA forecast saved to 'tsla_arima_forecast.csv' for Task 3.")