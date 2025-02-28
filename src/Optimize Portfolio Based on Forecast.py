# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from scipy.optimize import minimize

# Set plot style
plt.style.use('seaborn')

# Task 4: Optimize Portfolio Based on Forecast
# Step 1: Load historical data and TSLA forecast
historical_data = pd.read_csv('cleaned_adj_close.csv', index_col='Date', parse_dates=True)
tsla_forecast = pd.read_csv('tsla_future_forecast.csv', index_col='Date', parse_dates=True)['Forecast']

# Step 2: Forecast BND and SPY (using ARIMA as in Task 3)
assets = ['BND', 'SPY']
forecasts = {'TSLA': tsla_forecast}

for asset in assets:
    asset_data = historical_data[asset].dropna()
    # Train ARIMA with auto_arima for simplicity (can use Task 2 parameters if preferred)
    auto_model = pm.auto_arima(asset_data, start_p=1, start_q=1, max_p=3, max_q=3, 
                               d=None, seasonal=False, stepwise=True, trace=False,
                               error_action='ignore', suppress_warnings=True)
    arima_model = ARIMA(asset_data, order=auto_model.order)
    arima_fit = arima_model.fit()
    forecast_result = arima_fit.forecast(steps=252)  # 12 months
    forecast_dates = pd.date_range(start=asset_data.index[-1] + pd.Timedelta(days=1), 
                                   periods=252, freq='B')
    forecasts[asset] = pd.Series(forecast_result, index=forecast_dates)

# Step 3: Combine forecasts into one DataFrame
forecast_df = pd.DataFrame(forecasts)

# Step 4: Compute daily returns for the forecast period
forecast_returns = forecast_df.pct_change().dropna()

# Step 5: Portfolio optimization functions
def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252  # Annualized return

def portfolio_volatility(weights, returns):
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized volatility

def portfolio_sharpe(weights, returns, risk_free_rate=0.02):
    port_return = portfolio_return(weights, returns)
    port_vol = portfolio_volatility(weights, returns)
    return (port_return - risk_free_rate) / port_vol  # Sharpe Ratio

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
bounds = tuple((0, 1) for _ in range(3))  # Weights between 0 and 1
initial_weights = np.array([1/3, 1/3, 1/3])  # Equal allocation to start

# Step 6: Optimize portfolio to maximize Sharpe Ratio
result = minimize(lambda w: -portfolio_sharpe(w, forecast_returns), initial_weights,
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x
print("\n=== Optimal Portfolio Weights ===")
print(f"TSLA: {optimal_weights[0]:.4f}, BND: {optimal_weights[1]:.4f}, SPY: {optimal_weights[2]:.4f}")

# Step 7: Analyze Portfolio Risk and Return
port_return = portfolio_return(optimal_weights, forecast_returns)
port_vol = portfolio_volatility(optimal_weights, forecast_returns)
sharpe_ratio = portfolio_sharpe(optimal_weights, forecast_returns)

# Calculate VaR for TSLA (95% confidence)
tsla_returns = forecast_returns['TSLA']
var_95 = np.percentile(tsla_returns, 5) * 252  # Annualized VaR

print("\n=== Portfolio Risk and Return Analysis ===")
print(f"Expected Annual Return: {port_return:.4f} ({port_return*100:.2f}%)")
print(f"Portfolio Volatility: {port_vol:.4f} ({port_vol*100:.2f}%)")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"TSLA VaR (95% Confidence): {var_95:.4f} ({var_95*100:.2f}% potential loss)")

# Step 8: Simulate portfolio performance
portfolio_cum_returns = (forecast_returns @ optimal_weights + 1).cumprod()
plt.figure(figsize=(12, 6))
plt.plot(portfolio_cum_returns, label='Optimized Portfolio Cumulative Return', color='purple')
plt.title('Portfolio Performance Based on Forecasted Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Step 9: Risk-Return Scatter Plot (for intuition)
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
for i in range(num_portfolios):
    weights = np.random.random(3)
    weights /= np.sum(weights)
    results[0, i] = portfolio_return(weights, forecast_returns)
    results[1, i] = portfolio_volatility(weights, forecast_returns)
    results[2, i] = portfolio_sharpe(weights, forecast_returns)

plt.figure(figsize=(10, 6))
plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(port_vol, port_return, c='red', marker='*', s=200, label='Optimal Portfolio')
plt.title('Portfolio Risk-Return Analysis')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.legend()
plt.show()

# Step 10: Summarize and Recommend Adjustments
print("\n=== Portfolio Optimization Summary ===")
print(f"Optimal Weights: TSLA: {optimal_weights[0]*100:.2f}%, BND: {optimal_weights[1]*100:.2f}%, SPY: {optimal_weights[2]*100:.2f}%")
print(f"Expected Return: {port_return*100:.2f}%, Volatility: {port_vol*100:.2f}%, Sharpe: {sharpe_ratio:.4f}")
print("Adjustments Reasoning:")
if optimal_weights[0] > 0.5:
    print("- High TSLA allocation due to strong forecasted growth, accepting higher risk.")
elif optimal_weights[1] > 0.5:
    print("- High BND allocation to prioritize stability amid TSLA volatility.")
else:
    print("- Balanced SPY-heavy portfolio for diversification and moderate risk-return.")
if var_95 < -0.2:  # Arbitrary threshold for high risk
    print("- Caution: High TSLA VaR suggests potential for significant losses; consider reducing TSLA weight.")

# Save optimized portfolio data
portfolio_df = pd.DataFrame({
    'Weights': optimal_weights,
    'Assets': ['TSLA', 'BND', 'SPY']
}).set_index('Assets')
portfolio_df.to_csv('optimized_portfolio.csv')
print("\nOptimized portfolio saved to 'optimized_portfolio.csv'.")