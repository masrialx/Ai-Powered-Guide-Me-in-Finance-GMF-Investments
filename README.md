# GMF Investments: Task 1 - Preprocess and Explore Financial Data

## Overview

This project is part of the Guide Me in Finance (GMF) Investments initiative, focusing on preprocessing and exploring historical financial data to prepare it for time series forecasting and portfolio optimization. Task 1 involves loading, cleaning, and analyzing data for three assets: Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) using the `yfinance` library. The script conducts exploratory data analysis (EDA), examines trends, seasonality, volatility, and calculates key metrics like Value at Risk (VaR) and Sharpe Ratio.

### Objectives
- Fetch historical financial data (January 1, 2015, to January 31, 2025) for TSLA, BND, and SPY.
- Clean and preprocess the data (handle missing values, normalize if needed).
- Perform EDA to visualize trends, returns, and volatility.
- Decompose time series to analyze seasonality and trends.
- Document insights including VaR and Sharpe Ratio for risk and return assessment.

### Assets
- **TSLA**: High-growth, high-risk stock (Tesla, Inc.).
- **BND**: Stable, low-risk bond ETF (Vanguard Total Bond Market ETF).
- **SPY**: Diversified, moderate-risk ETF (S&P 500 Index).

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Pip**: Python package manager for installing dependencies.

## Setup Instructions

1. **Clone or Download the Repository**
   - Download the project files, including `task1.py`, `requirements.txt`, and this `README.md`.

2. **Install Dependencies**
   - Open a terminal or command prompt in the project directory.
   - Run the following command to install required libraries:
     ```bash
     pip install -r requirements.txt