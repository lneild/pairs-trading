import pandas as pd


def backtest(prices, signals, beta, tc=0.001):
    """
    Simulate P&L for a pairs-trading strategy:
      - `prices`: DataFrame with two columns [y, x]
      - `signals`: DataFrame with `position` column (1, -1, or 0)
      - `beta`: hedge ratio for sizing the second leg
      - `tc`: transaction cost per traded dollar volume
    Returns:
      - `returns`: Series of period P&L after costs
      - `cum_returns`: Series of cumulative P&L
    """
    # positions for y and x legs
    pos_y = signals['position']
    pos_x = -beta * pos_y

    # gross returns: yesterday's positions * today's pct change
    ret_y = pos_y.shift(1) * prices.iloc[:,0].pct_change()
    ret_x = pos_x.shift(1) * prices.iloc[:,1].pct_change()
    gross = ret_y + ret_x

    # transaction costs: dollar volume traded * tc
    trade_y = (pos_y.diff().abs() * prices.iloc[:,0])
    trade_x = (pos_x.diff().abs() * prices.iloc[:,1])
    cost    = (trade_y + trade_x) * tc

    # net returns and cumulative
    returns     = (gross - cost).fillna(0)
    cum_returns = returns.cumsum()
    return returns, cum_returns


if __name__ == '__main__':
    import yaml
    from strat import generate_signals
    from data_fetch import fetch_prices

    # Load config
    cfg    = yaml.safe_load(open('config.yml'))
    tickers = cfg['tickers']['pair']

    # Fetch prices and generate signals
    prices = fetch_prices(tickers, cfg['data']['start'], cfg['data']['end'], cfg['data']['interval'])
    signals, beta = generate_signals(prices,
                                     lookback=cfg['strategy']['lookback'],
                                     z_enter=cfg['strategy']['z_enter'],
                                     z_exit=cfg['strategy']['z_exit'])

    # Build position column
    pos, current = [], 0
    for _, row in signals.iterrows():
        if row['long']:
            current =  1
        elif row['short']:
            current = -1
        elif row['exit']:
            current =  0
        pos.append(current)
    signals['position'] = pos

    # Backtest
    returns, cum_returns = backtest(prices, signals, beta, tc=cfg['backtest']['tc_per_trade'])
    cum_returns.to_csv('cum_returns.csv')
    print(f"Final cumulative P&L: {cum_returns.iloc[-1]:.2f}")