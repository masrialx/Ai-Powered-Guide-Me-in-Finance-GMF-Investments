# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from datetime import datetime, timedelta

# Set plot style
plt.style.use('seaborn')

# Task 3: Forecast Future Market Trends
# Step 1: Load historical data (use full dataset for training)
data = pd.read_csv('cleaned_adj_close.csv', index_col='Date', parse_dates=True)
tsla_data = data['TSLA'].dropna()

# Step 2: Train ARIMA model on the full dataset
# Parameters from Task 2 (e.g., optimized via auto_arima; using p=2, d=1, q=2 as an example)
p, d, q = 2, 1, 2  # Replace with your optimized parameters from Task 2 if different
arima_model = ARIMA(tsla_data, order=(p, d, q))
arima_fit = arima_model.fit()

# Step 3: Forecast 12 months into the future (252 trading days)
forecast_steps = 252
forecast_result = arima_fit.get_forecast(steps=forecast_steps)

# Extract forecast mean and confidence intervals
forecast_mean = forecast_result.predicted_mean
confidence_intervals = forecast_result.conf_int(alpha=0.05)  # 95% confidence

# Generate future date index (assuming trading days)
last_date = tsla_data.index[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='B')  # 'B' for business days
forecast_series = pd.Series(forecast_mean.values, index=future_dates)
lower_ci = pd.Series(confidence_intervals['lower TSLA'].values, index=future_dates)
upper_ci = pd.Series(confidence_intervals['upper TSLA'].values, index=future_dates)

# Step 4: Visualize forecast alongside historical data
plt.figure(figsize=(12, 6))
plt.plot(tsla_data[-500:], label='Historical Data (Last 500 Days)', color='blue')  # Last ~2 years for clarity
plt.plot(forecast_series, label='Forecast (12 Months)', color='green')
plt.fill_between(future_dates, lower_ci, upper_ci, color='green', alpha=0.2, label='95% Confidence Interval')
plt.title('TSLA Stock Price Forecast (Next 12 Months)')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.show()

# Step 5: Interpret Results
# Trend Analysis
last_price = tsla_data[-1]
forecast_end = forecast_series[-1]
trend_direction = "upward" if forecast_end > last_price else "downward" if forecast_end < last_price else "stable"
trend_change = ((forecast_end - last_price) / last_price) * 100

# Volatility and Risk
ci_width = upper_ci - lower_ci
avg_ci_width = ci_width.mean()
volatility_increasing = ci_width[-1] > ci_width[0]

# Step 6: Print Insights
print("\n=== Forecast Analysis ===")
print("Trend Analysis:")
print(f"- Predicted trend: {trend_direction}")
print(f"- Percentage change over 12 months: {trend_change:.2f}%")
print(f"- Starting price: ${last_price:.2f}, Ending forecast price: ${forecast_end:.2f}")

print("\nVolatility and Risk:")
print(f"- Average confidence interval width: ${avg_ci_width:.2f}")
print(f"- Volatility trend: {'Increasing' if volatility_increasing else 'Stable or Decreasing'}")
print(f"- Maximum uncertainty (widest CI): ${ci_width.max():.2f} at {ci_width.idxmax().date()}")

print("\nMarket Opportunities and Risks:")
if trend_direction == "upward":
    print("- Opportunity: Potential price increase could yield high returns.")
elif trend_direction == "downward":
    print("- Risk: Expected price decline may suggest reducing exposure.")
else:
    print("- Stable: Low growth potential; consider diversification.")
if volatility_increasing:
    print("- Risk: Rising uncertainty may increase exposure to sudden price swings.")
else:
    print("- Opportunity: Stable volatility suggests predictable price movements.")

# Step 7: Save forecast data for Task 4
forecast_df = pd.DataFrame({
    'Forecast': forecast_series,
    'Lower_CI': lower_ci,
    'Upper_CI': upper_ci
})
forecast_df.to_csv('tsla_future_forecast.csv')
print("\nForecast data saved to 'tsla_future_forecast.csv' for Task 4.")

# Optional: Validate with latest real data (if available)
today = datetime.now().date()
if today > last_date.date():
    print("\nFetching latest TSLA data for comparison...")
    latest_data = yf.download('TSLA', start=last_date + timedelta(days=1), end=today, progress=False)
    if not latest_data.empty:
        latest_adj_close = latest_data['Adj Close']
        plt.figure(figsize=(12, 6))
        plt.plot(tsla_data[-100:], label='Historical Data', color='blue')
        plt.plot(latest_adj_close, label='Latest Real Data', color='orange')
        plt.plot(forecast_series[:len(latest_adj_close)], label='Forecast', color='green')
        plt.title('TSLA Forecast vs Latest Real Data')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price (USD)')
        plt.legend()
        plt.show()