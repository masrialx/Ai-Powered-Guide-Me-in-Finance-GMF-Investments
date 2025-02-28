# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

# Set the style for better visualizations
plt.style.use('seaborn')

# Task 1: Preprocess and Explore the Data
# Step 1: Load historical financial data using yfinance
tickers = ['TSLA', 'BND', 'SPY']
start_date = '2015-01-01'
end_date = '2025-01-31'

# Fetch data
data = yf.download(tickers, start=start_date, end=end_date, progress=False)

# Extract Adjusted Close prices for simplicity (accounts for dividends/splits)
adj_close = data['Adj Close']

# Step 2: Data Cleaning and Understanding
print("=== Basic Data Overview ===")
print(adj_close.head())  # Check the first few rows
print("\n=== Data Info ===")
print(adj_close.info())  # Check data types and missing values
print("\n=== Basic Statistics ===")
print(adj_close.describe())  # Summary statistics

# Check for missing values
print("\n=== Missing Values ===")
print(adj_close.isnull().sum())

# Handle missing values (interpolate for continuity in time series)
adj_close = adj_close.interpolate(method='linear')

# Verify no missing values remain
print("\n=== Missing Values After Interpolation ===")
print(adj_close.isnull().sum())

# Normalize data (Min-Max scaling) for potential ML models
normalized_data = (adj_close - adj_close.min()) / (adj_close.max() - adj_close.min())

# Step 3: Exploratory Data Analysis (EDA)
# Visualize Adjusted Closing Prices over time
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(adj_close.index, adj_close[ticker], label=ticker)
plt.title('Adjusted Closing Prices (2015-2025)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Calculate and plot daily percentage change (returns)
daily_returns = adj_close.pct_change().dropna()
plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(daily_returns.index, daily_returns[ticker], label=ticker, alpha=0.5)
plt.title('Daily Percentage Change (Returns)')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.show()

# Step 4: Analyze Volatility (Rolling Mean and Std Dev)
window = 30  # 30-day rolling window
rolling_mean = daily_returns.rolling(window=window).mean()
rolling_std = daily_returns.rolling(window=window).std()

plt.figure(figsize=(12, 6))
for ticker in tickers:
    plt.plot(rolling_std.index, rolling_std[ticker], label=f'{ticker} Volatility')
plt.title(f'30-Day Rolling Standard Deviation (Volatility)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()

# Step 5: Outlier Detection (Days with unusually high/low returns)
threshold = 3  # Z-score threshold for outliers
outliers = {}
for ticker in tickers:
    z_scores = np.abs((daily_returns[ticker] - daily_returns[ticker].mean()) / daily_returns[ticker].std())
    outliers[ticker] = daily_returns[ticker][z_scores > threshold]

print("\n=== Outliers (High/Low Returns) ===")
for ticker in tickers:
    print(f"\n{ticker} Outliers:")
    print(outliers[ticker])

# Step 6: Seasonality and Trends (Decompose TSLA Time Series)
# Decompose TSLA Adj Close (assuming additive model)
decomposition = seasonal_decompose(adj_close['TSLA'].dropna(), model='additive', period=252)  # Annual seasonality (252 trading days)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.suptitle('TSLA Time Series Decomposition (Trend, Seasonal, Residual)')
plt.show()

# Step 7: Risk and Performance Metrics
# Calculate Value at Risk (VaR) at 95% confidence level
var_level = 0.05
var = daily_returns.quantile(var_level)
print("\n=== Value at Risk (VaR) at 95% Confidence ===")
for ticker in tickers:
    print(f"{ticker}: {var[ticker]:.4f} (Potential loss of {abs(var[ticker]*100):.2f}%)")

# Calculate Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized
print("\n=== Annualized Sharpe Ratio ===")
for ticker in tickers:
    print(f"{ticker}: {sharpe_ratio[ticker]:.4f}")

# Step 8: Document Key Insights
print("\n=== Key Insights ===")
print("1. TSLA Trend: Highly volatile with significant upward growth over the period.")
print("2. BND Stability: Low volatility, providing portfolio stability.")
print("3. SPY Exposure: Moderate volatility with steady growth reflecting broad market trends.")
print("4. Volatility Peaks: TSLA shows higher rolling std dev, indicating greater risk.")
print(f"5. VaR Analysis: TSLA has higher potential loss ({abs(var['TSLA']*100):.2f}%) compared to BND and SPY.")
print(f"6. Sharpe Ratio: TSLA ({sharpe_ratio['TSLA']:.4f}) offers high risk-adjusted returns, BND lower but stable.")

# Save cleaned data for future tasks
adj_close.to_csv('cleaned_adj_close.csv')
daily_returns.to_csv('daily_returns.csv')

print("\nData preprocessing and EDA completed. Cleaned data saved as CSV files.")