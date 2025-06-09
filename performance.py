import numpy as np
import matplotlib.pyplot as plt


def sharpe(returns, freq=252):
    """Annualized Sharpe ratio of return Series"""
    return np.sqrt(freq) * returns.mean() / returns.std()


def max_drawdown(cum_returns):
    """Worst peak-to-trough drawdown"""
    peak    = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()


def report_performance(returns, cum_returns):
    print(f"Total Return:   {cum_returns.iloc[-1]:.2f}")
    print(f"Sharpe Ratio:   {sharpe(returns):.2f}")
    print(f"Max Drawdown:   {max_drawdown(cum_returns):.2%}")

    plt.figure()
    plt.plot(cum_returns)
    plt.title("Cumulative P&L")
    plt.xlabel("Date")
    plt.ylabel("P&L")
    plt.show()


if __name__ == '__main__':
    import yaml
    import pandas as pd
    from backtest import backtest
    from strat import generate_signals
    from data_fetch import fetch_prices

    cfg     = yaml.safe_load(open('config.yml'))
    tickers = cfg['tickers']['pair']

    prices = fetch_prices(tickers, cfg['data']['start'], cfg['data']['end'], cfg['data']['interval'])
    signals, beta = generate_signals(prices,
                                     lookback=cfg['strategy']['lookback'],
                                     z_enter=cfg['strategy']['z_enter'],
                                     z_exit=cfg['strategy']['z_exit'])

    # build position in the same way
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

    returns, cum_returns = backtest(prices, signals, beta, tc=cfg['backtest']['tc_per_trade'])
    report_performance(returns, cum_returns)

