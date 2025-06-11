import yaml
import yfinance as yf
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from strat import hedge_ratio
import numpy as np

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
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    return prices.dropna(axis=1)

def spread_half_life(spread):
    """Estimate half-life of mean reversion for a spread."""
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    spread_lag = spread_lag - spread_lag.mean()  # Center the lagged spread
    model = OLS(spread_diff, spread_lag, missing='drop').fit()
    phi = model.params.iloc[0]
    if phi < 0:  # Negative phi indicates mean reversion
        half_life = -np.log(2) / np.log(1 + phi)
        return half_life if half_life > 0 else np.inf
    return np.inf

def find_pairs(prices, p_thresh, corr_thresh, vol_thresh):
    """
    Test every unique pair of columns in `prices` for cointegration.
    Returns:
      - pairs: list of (x, y) tuples ranked by composite score
      - scores: dict mapping (x, y) -> (p-value, spread_vol, half_life, score)
    """
    pairs, scores = [], {}
    cols = prices.columns
    vol_data = yf.download(cols, start=prices.index[0], end=prices.index[-1], progress=False, auto_adjust=True)['Volume']
    for i, x in enumerate(cols):
        for y in cols[i+1:]:
            # Cointegration test
            _, p, _ = coint(prices[x], prices[y])
            if p < p_thresh:
                # Correlation check
                corr = prices[x].corr(prices[y])
                # Liquidity check
                avg_vol_x = vol_data[x].mean()
                avg_vol_y = vol_data[y].mean()
                # Stationarity and spread stats
                beta = hedge_ratio(prices[y], prices[x])
                spread = prices[y] - beta * prices[x]
                adf_p = adfuller(spread.dropna())[1]
                spread_vol = spread.rolling(20).std().mean()  # Avg spread volatility
                half_life = spread_half_life(spread.dropna())
                if (corr > corr_thresh and avg_vol_x > vol_thresh and 
                    avg_vol_y > vol_thresh and adf_p < 0.05):
                    # Composite score: lower p-value, higher vol, faster reversion
                    score = (1 - p / p_thresh) + (spread_vol / 0.05) - (half_life / 20)
                    pairs.append((x, y))
                    scores[(x, y)] = (p, spread_vol, half_life, score)
    # Sort by composite score (higher is better)
    pairs.sort(key=lambda pair: scores[pair][3], reverse=True)
    return pairs, scores

if __name__ == '__main__':
    cfg = load_cfg()
    uni = fetch_universe(
        cfg['tickers']['universe'],
        cfg['data']['start'],
        cfg['data']['end']
    )
    p_thresh = cfg['pair_selection']['p_thresh']
    corr_thresh = cfg['pair_selection']['corr_thresh']
    vol_thresh = cfg['pair_selection']['vol_thresh']
    pairs, scores = find_pairs(uni, p_thresh=p_thresh, corr_thresh=corr_thresh, vol_thresh=vol_thresh)
    df = pd.DataFrame(
        [{'pair': f"{x},{y}", 'p_value': p, 'spread_vol': vol, 'half_life': hl, 'score': s}
         for (x, y), (p, vol, hl, s) in scores.items()]
    )
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    print("\nAll cointegrated pairs sorted by composite score:")
    print(df.to_markdown())