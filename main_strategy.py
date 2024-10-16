import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import yfinance as yf
import pandas_datareader as pdr
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import pytz
from datetime import datetime, timedelta
from backtest import backtest, ensure_utc
import logging
from portfolio_manager import PortfolioManager
from dotenv import load_dotenv
import os

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Alpaca API client
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_API_SECRET")
api = tradeapi.REST(API_KEY, SECRET_KEY, base_url='https://paper-api.alpaca.markets')
portfolio_manager = PortfolioManager(API_KEY, SECRET_KEY, paper=True)

def get_sp500_tickers():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500['Symbol'].tolist()

def safe_download(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    return data.dropna(axis=1, how='all')

def calculate_features(price_data, volume_data):
    features = pd.DataFrame(index=price_data.index)
    for column in price_data.columns:
        column_data = price_data[column].dropna()
        volume_column_data = volume_data[column].dropna()
        if not column_data.empty and not volume_column_data.empty:
            features[f'{column}_returns'] = column_data.pct_change(fill_method=None)
            features[f'{column}_volatility'] = features[f'{column}_returns'].rolling(window=20).std()
            features[f'{column}_momentum'] = column_data.pct_change(periods=20, fill_method=None)
            features[f'{column}_volume'] = volume_column_data
    return features.dropna()

def calculate_returns(data, periods):
    returns = pd.DataFrame()
    for period in periods:
        period_returns = data.pct_change(periods=period, fill_method=None)
        returns = pd.concat([returns, period_returns.add_suffix(f'_{period}M')], axis=1)
    return returns

def get_fama_french_factors(start_date, end_date):
    ff_factors = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3_daily', start=start_date, end=end_date)[0]
    ff_factors = ff_factors.div(100)  # Convert to decimal format
    ff_factors.index = pd.to_datetime(ff_factors.index)
    ff_factors.index = ff_factors.index.tz_localize('UTC')
    return ff_factors

def calculate_rolling_factor_betas(returns, factors, min_window=20):
    returns.index = returns.index.tz_convert('UTC')
    factors.index = factors.index.tz_convert('UTC')
    
    returns = returns[~returns.index.duplicated(keep='first')]
    factors = factors[~factors.index.duplicated(keep='first')]
    
    common_dates = returns.index.intersection(factors.index)
    returns = returns.loc[common_dates]
    factors = factors.reindex(common_dates)
    
    window = min(len(returns), max(min_window, len(returns) // 10))
    
    print(f"Using rolling window of {window} days")
    
    betas = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    for ticker in returns.columns:
        Y = returns[ticker] - factors['RF']
        X = sm.add_constant(factors.drop('RF', axis=1))
        
        try:
            rols = RollingOLS(Y, X, window=window)
            rolling_res = rols.fit()
            
            for factor in X.columns:
                if factor != 'const':
                    betas[f'{ticker}_{factor}'] = rolling_res.params[factor]
        except Exception as e:
            print(f"Error fitting RollingOLS for {ticker}: {e}")
    
    return betas

def cluster_assets(features, n_clusters=5):
    if len(features) < n_clusters:
        return pd.Series(range(len(features)), index=features.index)
    
    features_normalized = (features - features.mean()) / features.std()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_normalized)
    return pd.Series(clusters, index=features.index)

def portfolio_optimization(returns, risk_free_rate=0.02):
    if returns.empty:
        return pd.Series()
    def objective(weights, returns):
        portfolio_return = np.sum(returns.mean() * weights) * 12
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 12, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(len(returns.columns)))
    
    try:
        result = minimize(objective, len(returns.columns) * [1./len(returns.columns),], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)
        return pd.Series(result.x, index=returns.columns)
    except:
        return pd.Series(1/len(returns.columns), index=returns.columns)

def main(lookback_years=10):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*lookback_years)).strftime('%Y-%m-%d')
    
    logging.info(f"Fetching data from {start_date} to {end_date}")
    
    tickers = get_sp500_tickers()
    data = safe_download(tickers, start=start_date, end=end_date)

    data.index = data.index.tz_convert('UTC')
    data = data[~data.index.duplicated(keep='first')]

    features = calculate_features(data['Adj Close'], data['Volume'])

    monthly_data = data['Adj Close'].resample('ME').last()
    monthly_volume = data['Volume'].resample('ME').sum()
    liquidity = monthly_volume.mean()
    top_150_liquid = liquidity.nlargest(150).index.tolist()

    monthly_returns = calculate_returns(monthly_data[top_150_liquid], [1, 3, 6, 12])

    ff_factors = get_fama_french_factors(start_date, end_date)
    ff_factors = ff_factors[~ff_factors.index.duplicated(keep='first')]

    daily_returns = data['Adj Close'].pct_change().dropna()

    try:
        rolling_betas = calculate_rolling_factor_betas(daily_returns, ff_factors)
        logging.info("Successfully calculated rolling betas")
    except Exception as e:
        logging.error(f"Error in calculate_rolling_factor_betas: {e}")
        return

    monthly_betas = rolling_betas.resample('ME').last()

    features = pd.concat([features, monthly_betas], axis=1)

    monthly_clusters = features.groupby(pd.Grouper(freq='ME')).apply(cluster_assets)

    monthly_portfolios = monthly_returns.filter(regex='_1M$').groupby(pd.Grouper(freq='ME')).apply(portfolio_optimization)

    # Get the latest portfolio allocation
    latest_portfolio = monthly_portfolios.iloc[-1]

    # Update the Alpaca portfolio using the PortfolioManager
    portfolio_manager.update_portfolio(latest_portfolio)

    return monthly_returns, monthly_portfolios, start_date, end_date

if __name__ == "__main__":
    monthly_returns, monthly_portfolios, start_date, end_date = main()
    logging.info(f"Strategy executed from {start_date} to {end_date}")

    # Schedule the next run
    import schedule
    import time

    def job():
        monthly_returns, monthly_portfolios, start_date, end_date = main()
        logging.info(f"Strategy executed from {start_date} to {end_date}")

    # Schedule the job to run on the first day of each month at 00:00
    schedule.every().month.at("00:00").do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)