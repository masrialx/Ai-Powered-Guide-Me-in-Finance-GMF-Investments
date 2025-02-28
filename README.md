# GMF Investments: Portfolio Forecasting and Optimization

## Overview

This project, developed for Guide Me in Finance (GMF) Investments, leverages time series forecasting and portfolio optimization to enhance investment strategies. It focuses on three assets: Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY). The project is divided into four tasks:

1. **Task 1**: Preprocess and explore historical financial data.
2. **Task 2**: Develop a time series forecasting model for TSLA stock prices.
3. **Task 3**: Forecast future TSLA prices and analyze trends.
4. **Task 4**: Optimize a portfolio using forecasts for TSLA, BND, and SPY.

### Assets
- **TSLA**: High-growth, high-risk stock.
- **BND**: Stable, low-risk bond ETF.
- **SPY**: Diversified, moderate-risk index fund.

### Objectives
- Equip users with skills in data preprocessing, time series forecasting, and portfolio optimization.
- Use data-driven insights to maximize returns and minimize risks.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Pip**: Python package manager.
- **Git**: For version control (optional).

## Setup Instructions

1. **Clone or Download the Repository**
   ```bash
   git clone <repository-url>
   cd gmf-investments
   ```

2. **Install Dependencies**
   - Use the provided `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```
   - Contents of `requirements.txt`:
     ```
     yfinance>=0.2.40
     pandas>=2.0.0
     numpy>=1.24.0
     matplotlib>=3.7.0
     statsmodels>=0.14.0
     seaborn>=0.12.0
     pmdarima>=2.0.4
     scikit-learn>=1.2.0
     scipy>=1.10.0
     # tensorflow>=2.10.0  # Uncomment for LSTM in Task 2
     ```

3. **Verify Installation**
   ```bash
   python -c "import yfinance, pandas, numpy, matplotlib, statsmodels, seaborn, pmdarima, sklearn, scipy"
   ```

## Tasks

### Task 1: Preprocess and Explore the Data

#### Description
Loads, cleans, and analyzes historical data for TSLA, BND, and SPY from January 1, 2015, to January 31, 2025 using `yfinance`.

#### Usage
```bash
python task1.py
```

#### Outputs
- Visualizations: Closing prices, daily returns, rolling volatility.
- Metrics: VaR, Sharpe Ratio, outliers.
- Files: `cleaned_adj_close.csv`, `daily_returns.csv`.

#### Insights
- TSLA: High volatility with upward trend.
- BND: Low volatility, stable.
- SPY: Moderate risk, broad market exposure.

---

### Task 2: Develop Time Series Forecasting Models

#### Description
Builds an ARIMA model to forecast TSLA stock prices, with options for SARIMA or LSTM (commented out).



#### Outputs
- Forecast plot comparing test data.
- Metrics: MAE, RMSE, MAPE.
- File: `tsla_arima_forecast.csv`.

#### Insights
- ARIMA effectively captures short-term trends.
- Evaluation metrics guide model performance.

---

### Task 3: Forecast Future Market Trends

#### Description
Uses the ARIMA model to forecast TSLA prices for 12 months, including confidence intervals and trend analysis.



#### Outputs
- Forecast plot with historical data and confidence intervals.
- Analysis: Trends, volatility, opportunities, risks.
- File: `tsla_future_forecast.csv`.

#### Insights
- Trend direction (e.g., upward/downward) informs investment strategy.
- Confidence intervals highlight uncertainty.

---

### Task 4: Optimize Portfolio Based on Forecast

#### Description
Forecasts BND and SPY, combines with TSLA forecast, and optimizes a portfolio to maximize the Sharpe Ratio.



#### Outputs
- Plots: Cumulative returns, risk-return scatter.
- Metrics: Return, volatility, Sharpe Ratio, TSLA VaR.
- File: `optimized_portfolio.csv`.

#### Insights
- Optimal weights balance TSLA’s growth with BND’s stability and SPY’s diversification.
- High TSLA VaR may suggest reducing exposure if risk tolerance is low.

## File Structure

- `task1.py`: Preprocessing and EDA.
- `task2.py`: Forecasting model development.
- `task3.py`: Future trend forecasting.
- `task4.py`: Portfolio optimization.
- `requirements.txt`: Dependencies.
- `README.md`: This documentation.
- **Generated Files**:
  - `cleaned_adj_close.csv`, `daily_returns.csv` (Task 1)
  - `tsla_arima_forecast.csv` (Task 2)
  - `tsla_future_forecast.csv` (Task 3)
  - `optimized_portfolio.csv` (Task 4)

## Usage Notes

1. **Run Sequentially**: Tasks depend on prior outputs (e.g., Task 2 needs Task 1’s CSV).
2. **Customization**:
   - Adjust ARIMA parameters in `task2.py` and `task3.py` based on `auto_arima` results.
   - Modify risk-free rate or optimization goals in `task4.py`.
3. **Date Limitation**: Data is current to February 28, 2025; forecasts extend from the last historical date.

## Troubleshooting

- **Missing Files**: Ensure prior tasks are run to generate CSVs.
- **yfinance Errors**: Check internet connection or reduce date range.
- **Plot Issues**: Use `plt.switch_backend('TkAgg')` if plots don’t display.

## Key Insights

- **TSLA**: Offers high returns with significant risk.
- **BND**: Stabilizes the portfolio.
- **SPY**: Provides balanced exposure.
- **Portfolio**: Optimized weights reflect risk tolerance and market outlook.

## Next Steps

- Extend forecasting to other models (e.g., LSTM) for improved accuracy.
- Incorporate real-time data updates.
- Expand portfolio to additional assets.

