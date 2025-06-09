import yaml
import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint


def load_cfg(path='config.yml'):
    return yaml.safe_load(open(path))


def fetch_universe(tickers, start, end):
    """
    Download price series for your universe.
    If 'Adj Close' exists, use it; otherwise fall back to the adjusted 'Close'.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    # pick the right price series
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']

    # drop any ticker with missing days
    return prices.dropna(axis=1)


def find_pairs(prices, p_thresh):
    """
    Test every unique pair of columns in `prices` for cointegration.
    Returns:
      - pairs: list of (x, y) tuples with p-value < p_thresh
      - pvals: dict mapping (x, y) -> p-value
    """
    pairs, pvals = [], {}
    cols = prices.columns
    for i, x in enumerate(cols):
        for y in cols[i+1:]:
            _, p, _ = coint(prices[x], prices[y])
            if p < p_thresh:
                pairs.append((x, y))
                pvals[(x, y)] = p
    # sort by p-value ascending
    pairs.sort(key=lambda pair: pvals[pair])
    return pairs, pvals


if __name__ == '__main__':
    # 1. Load config and universe
    cfg     = load_cfg()
    uni     = fetch_universe(
                  cfg['tickers']['universe'],
                  cfg['data']['start'],
                  cfg['data']['end']
              )

    # grab threshold and find pairs
    p_thresh = cfg['pair_selection']['p_thresh']
    pairs, pvals = find_pairs(uni, p_thresh=p_thresh)

    # df of all pairs sorted by p-value
    df = pd.DataFrame(
        [{'pair': f"{x},{y}", 'p_value': p} for (x, y), p in pvals.items()]
    )
    df = df.sort_values('p_value').reset_index(drop=True)

    print("\nAll cointegrated pairs sorted by p-value:")
    print(df.to_markdown())
