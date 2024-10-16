import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

def ensure_utc(index):
    if index.tz is None:
        return index.tz_localize('UTC')
    return index.tz_convert('UTC')

def backtest(data, weights):
    if data.empty or weights.empty:
        print("Error: Empty data or weights in backtest function")
        return pd.Series()
    
    aligned_data, aligned_weights = data.align(weights, join='inner', axis=0)
    
    if aligned_data.empty or aligned_weights.empty:
        print("Error: No overlap between data and weights dates")
        return pd.Series()
    
    portfolio_returns = (aligned_data * aligned_weights.shift(1)).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns

def run_backtest(monthly_returns, monthly_portfolios, start_date, end_date):
    portfolio_performance = backtest(monthly_returns.filter(regex='_1M$'), monthly_portfolios)
    
    portfolio_performance.index = ensure_utc(portfolio_performance.index)
    
    print(f"Portfolio performance shape: {portfolio_performance.shape}")
    print(f"Portfolio performance date range: {portfolio_performance.index.min()} to {portfolio_performance.index.max()}")

    if portfolio_performance.empty:
        print("Error: Portfolio performance is empty")
        return

    # Download SPY data for comparison
    spy_data = yf.download('SPY', start=start_date, end=end_date)['Adj Close']
    spy_monthly_returns = spy_data.resample('M').last().pct_change()
    spy_monthly_returns.index = ensure_utc(spy_monthly_returns.index)

    print(f"SPY monthly returns shape: {spy_monthly_returns.shape}")
    print(f"SPY monthly returns date range: {spy_monthly_returns.index.min()} to {spy_monthly_returns.index.max()}")

    # Align data
    portfolio_performance = portfolio_performance.sort_index()
    spy_monthly_returns = spy_monthly_returns.sort_index()

    start_date = max(portfolio_performance.index.min(), spy_monthly_returns.index.min())
    end_date = min(portfolio_performance.index.max(), spy_monthly_returns.index.max())

    print(f"Common date range: {start_date} to {end_date}")

    portfolio_performance = portfolio_performance.loc[start_date:end_date]
    spy_performance = (1 + spy_monthly_returns.loc[start_date:end_date]).cumprod()

    print(f"Aligned portfolio performance shape: {portfolio_performance.shape}")
    print(f"Aligned SPY performance shape: {spy_performance.shape}")

    if portfolio_performance.empty or spy_performance.empty:
        print("Error: Portfolio performance or SPY performance is empty after alignment")
        return

    portfolio_performance = portfolio_performance.asfreq('ME')
    spy_performance = spy_performance.asfreq('ME')

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_performance.index, portfolio_performance.values, label='Portfolio')
    plt.plot(spy_performance.index, spy_performance.values, label='S&P 500 (SPY)')
    plt.legend()
    plt.title(f'Portfolio Performance vs S&P 500 ({start_date.date()} to {end_date.date()})')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()

    # Calculate and print performance metrics
    portfolio_total_return = portfolio_performance.values[-1] - 1
    spy_total_return = spy_performance.values[-1] - 1

    print(f"Portfolio Total Return: {portfolio_total_return:.2%}")
    print(f"S&P 500 Total Return: {spy_total_return:.2%}")

    years = len(portfolio_performance) / 12  # Assuming monthly data
    portfolio_annualized_return = np.power(portfolio_total_return + 1, 1 / years) - 1
    spy_annualized_return = np.power(spy_total_return + 1, 1 / years) - 1

    print(f"Portfolio Annualized Return: {portfolio_annualized_return:.2%}")
    print(f"S&P 500 Annualized Return: {spy_annualized_return:.2%}")

if __name__ == "__main__":
    from main_strategy import main
    monthly_returns, monthly_portfolios, start_date, end_date = main()
    run_backtest(monthly_returns, monthly_portfolios, start_date, end_date)