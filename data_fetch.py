import yfinance as yf
import pandas as pd

def fetch_prices(tickers, start, end, interval='1d'):
    """
    Download the adjusted price series for your selected pair.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    # same logic as in pair_selection.py
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']

    return prices.dropna()

if __name__ == '__main__':
    import yaml

    cfg     = yaml.safe_load(open('config.yml'))
    tickers = cfg['tickers']['pair']
    start   = cfg['data']['start']
    end     = cfg['data']['end']

    prices = fetch_prices(tickers, start, end)
    prices.to_csv('prices.csv')
    print(prices.tail().to_markdown())
