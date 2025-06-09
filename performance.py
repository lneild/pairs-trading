import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sharpe(returns, freq=252):
    """Annualized Sharpe ratio of daily return Series"""
    return np.sqrt(freq) * returns.mean() / returns.std()


def max_drawdown(cum_returns):
    """Worst peak-to-trough drawdown"""
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()


def trade_stats(returns, signals):
    """
    Compute basic per-trade statistics:
      - number of trades
      - win rate
      - average P&L per trade
    """
    df = pd.DataFrame({'returns': returns, 'position': signals['position']})
    # identify periods in trade
    df['in_trade'] = df['position'] != 0
    # mark new trades when in_trade starts
    df['trade_id'] = (df['in_trade'] & ~df['in_trade'].shift(1).fillna(False)).cumsum()
    # filter only periods within trades
    trade_returns = df[df['in_trade']].groupby('trade_id')['returns'].sum()
    num_trades = trade_returns.shape[0]
    win_rate = (trade_returns > 0).mean() if num_trades > 0 else np.nan
    avg_pnl = trade_returns.mean() if num_trades > 0 else np.nan
    return num_trades, win_rate, avg_pnl


def report_performance(returns, cum_returns, signals=None):
    """
    Print key performance metrics and plot equity curve.
    If `signals` is provided, also prints trade-level stats.
    """
    print(f"Total Return:     {cum_returns.iloc[-1]:.2f}")
    print(f"Sharpe Ratio:     {sharpe(returns):.2f}")
    print(f"Max Drawdown:    {max_drawdown(cum_returns):.2%}")

    # trade-level stats
    if signals is not None and 'position' in signals:
        num_trades, win_rate, avg_pnl = trade_stats(returns, signals)
        print(f"Number of trades: {num_trades}")
        print(f"Win rate:         {win_rate:.2%}")
        print(f"Avg P&L/trade:    {avg_pnl:.2f}")

    # plot equity curve
    plt.figure()
    plt.plot(cum_returns)
    plt.title("Cumulative P&L")
    plt.xlabel("Date")
    plt.ylabel("P&L")
    plt.show()

if __name__ == '__main__':
    import yaml
    from backtest import backtest
    from strat import generate_signals
    from data_fetch import fetch_prices

    # load config
    cfg = yaml.safe_load(open('config.yml'))
    tickers = cfg['tickers']['pair']
    # fetch data and generate signals
    prices = fetch_prices(tickers, cfg['data']['start'], cfg['data']['end'], cfg['data'].get('interval', '1d'))
    signals, beta = generate_signals(
        prices,
        lookback=cfg['strategy']['lookback'],
        z_enter=cfg['strategy']['z_enter'],
        z_exit=cfg['strategy']['z_exit']
    )
    # build position
    pos = []
    current = 0
    for _, row in signals.iterrows():
        if row['long']:
            current =  1
        elif row['short']:
            current = -1
        elif row['exit']:
            current =  0
        pos.append(current)
    signals['position'] = pos

    # backtest
    returns, cum_returns = backtest(
        prices,
        signals,
        beta,
        tc=cfg['backtest']['tc_per_trade']
    )
    # report performance including trade stats
    report_performance(returns, cum_returns, signals)
