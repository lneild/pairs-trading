# strategy.py

import yaml
import pandas as pd
from statsmodels.regression.linear_model import OLS

def load_cfg(path='config.yml'):
    """Read YAML config into a dict."""
    return yaml.safe_load(open(path))

def hedge_ratio(y, x):
    """
    Compute the regression slope β of y ~ x
    to use as the hedge ratio.
    """
    model = OLS(y, x).fit()
    return model.params.iloc[0]

def generate_signals(prices, lookback, z_enter, z_exit):
    """
    Given a two-column price DataFrame:
      - Compute β
      - Compute spread = y - β x
      - Compute rolling z-score of spread
      - Signal long/short when |z| > z_enter
      - Signal exit when |z| < z_exit
    Returns (signals_df, beta).
    """
    y, x = prices.iloc[:,0], prices.iloc[:,1]
    β    = hedge_ratio(y, x)
    spread = y - β * x

    μ = spread.rolling(lookback).mean()
    σ = spread.rolling(lookback).std()
    z = (spread - μ) / σ

    sig = pd.DataFrame(index=prices.index)
    sig['long']  = (z < -z_enter).astype(int)
    sig['short'] = (z >  z_enter).astype(int)
    sig['exit']  = (z.abs() < z_exit).astype(int)

    return sig, β

if __name__ == '__main__':
    # Load config and prices
    cfg    = load_cfg()
    prices = pd.read_csv('prices.csv', index_col=0, parse_dates=True)

    # Generate signals and hedge ratio
    signals, beta = generate_signals(
        prices,
        lookback=cfg['strategy']['lookback'],
        z_enter=cfg['strategy']['z_enter'],
        z_exit=cfg['strategy']['z_exit']
    )
    # optional: persist raw signals
    signals.to_csv('signals.csv')

    # Build a running position: 1=long, -1=short, 0=flat
    position = []
    current = 0
    for _, row in signals.iterrows():
        if row['long']:
            current =  1
        elif row['short']:
            current = -1
        elif row['exit']:
            current =  0
        position.append(current)
    signals['position'] = position

    # Filter to only entry/exit events (position changes)
    events = signals[signals['position'].diff().fillna(0) != 0]

    # Display the trade events and hedge ratio
    print("\nTrade events (entries & exits):")
    print(events[['long','short','exit','position']].to_markdown())
    print(f"\nHedge ratio β = {beta:.4f}")
