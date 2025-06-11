import pandas as pd

def backtest(prices, signals, beta, tc=0.001, stop_loss=-0.05):
    """
    Simulate P&L for a pairs-trading strategy:
      - `prices`: DataFrame with two columns [y, x]
      - `signals`: DataFrame with `position` column (1, -1, or 0)
      - `beta`: hedge ratio for sizing the second leg
      - `tc`: transaction cost per traded dollar volume
      - `stop_loss`: exit if trade P&L drops below this threshold
    Returns:
      - `returns`: Series of period P&L after costs
      - `cum_returns`: Series of cumulative P&L
    """
    # Compute spread volatility for position sizing
    spread = prices.iloc[:,0] - beta * prices.iloc[:,1]
    vol = spread.rolling(20).std()
    inv_vol = 1 / vol  # Inverse volatility for sizing
    pos_y = signals['position'] * inv_vol
    pos_x = -beta * pos_y

    # Gross returns
    ret_y = pos_y.shift(1) * prices.iloc[:,0].pct_change()
    ret_x = pos_x.shift(1) * prices.iloc[:,1].pct_change()
    gross = ret_y + ret_x

    # Transaction costs
    trade_y = (pos_y.diff().abs() * prices.iloc[:,0])
    trade_x = (pos_x.diff().abs() * prices.iloc[:,1])
    cost = (trade_y + trade_x) * tc

    # Net returns
    returns = (gross - cost).fillna(0)
    
    # Apply stop-loss
    cum_trade = 0
    for i, r in returns.items():
        if signals['position'][i] != 0:
            cum_trade += r
            if cum_trade < stop_loss:
                signals.at[i, 'position'] = 0  # Exit trade
                cum_trade = 0
        else:
            cum_trade = 0
    # Recalculate after stop-loss
    pos_y = signals['position'] * inv_vol
    pos_x = -beta * pos_y
    ret_y = pos_y.shift(1) * prices.iloc[:,0].pct_change()
    ret_x = pos_x.shift(1) * prices.iloc[:,1].pct_change()
    gross = ret_y + ret_x
    trade_y = (pos_y.diff().abs() * prices.iloc[:,0])
    trade_x = (pos_x.diff().abs() * prices.iloc[:,1])
    cost = (trade_y + trade_x) * tc
    returns = (gross - cost).fillna(0)
    cum_returns = returns.cumsum()
    return returns, cum_returns

if __name__ == '__main__':
    import yaml
    from strat import generate_signals
    from data_fetch import fetch_prices

    cfg = yaml.safe_load(open('config.yml'))
    tickers = cfg['tickers']['pair']
    prices = fetch_prices(tickers, cfg['data']['start'], cfg['data']['end'], cfg['data']['interval'])
    signals, beta = generate_signals(
        prices,
        lookback=cfg['strategy']['lookback'],
        z_enter=cfg['strategy']['z_enter'],
        z_exit=cfg['strategy']['z_exit']
    )
    pos, current = [], 0
    for _, row in signals.iterrows():
        if row['long']:
            current = 1
        elif row['short']:
            current = -1
        elif row['exit']:
            current = 0
        pos.append(current)
    signals['position'] = pos
    returns, cum_returns = backtest(prices, signals, beta, tc=cfg['backtest']['tc_per_trade'])
    cum_returns.to_csv('cum_returns.csv')
    print(f"Final cumulative P&L: {cum_returns.iloc[-1]:.2f}")