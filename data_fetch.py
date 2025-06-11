import yfinance as yf
import pandas as pd

def fetch_prices(tickers, start, end, interval='1d', min_vol=100000):
    """
    Download the adjusted price series for your selected pair.
    Filters out extreme returns and low-volume days.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False
    )
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    # Fetch volume
    vol = yf.download(tickers, start=start, end=end, interval=interval, progress=False, auto_adjust=False)['Volume']
    # Filter low volume days
    vol_mask = (vol >= min_vol).all(axis=1)
    prices = prices[vol_mask]
    # Filter extreme returns (>50%)
    returns = prices.pct_change()
    extreme_mask = (returns.abs() < 0.5).all(axis=1)
    prices = prices[extreme_mask].dropna()
    return prices

if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open('config.yml'))
    tickers = cfg['tickers']['pair']
    start = cfg['data']['start']
    end = cfg['data']['end']
    prices = fetch_prices(tickers, start, end, cfg['data']['interval'])
    prices.to_csv('prices.csv')
    print(prices.tail().to_markdown())