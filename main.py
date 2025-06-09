import yaml
from data_fetch import fetch_prices
from pair_selection import fetch_universe, find_pairs
from strat import generate_signals
from backtest import backtest
from performance import report_performance

def load_cfg(path='config.yml'):
    return yaml.safe_load(open(path))

def main():
    cfg = load_cfg()

    # Find cointegrated pair
    uni       = fetch_universe(cfg['tickers']['universe'],
                               cfg['data']['start'],
                               cfg['data']['end'])
    p_thresh  = cfg['pair_selection']['p_thresh']
    pairs, _  = find_pairs(uni, p_thresh=p_thresh)
    t1, t2    = pairs[0] if pairs else cfg['tickers']['pair']

    # Fetch price data
    prices = fetch_prices([t1, t2],
                          cfg['data']['start'],
                          cfg['data']['end'],
                          cfg['data']['interval'])

    # Generate trading signals
    signals, beta = generate_signals(
        prices,
        lookback=cfg['strategy']['lookback'],
        z_enter= cfg['strategy']['z_enter'],
        z_exit=  cfg['strategy']['z_exit']
    )

    # Filter: minimum spread volatility
    spread = prices.iloc[:,0] - beta * prices.iloc[:,1]
    vol    = spread.rolling(cfg['strategy']['lookback']).std()
    low_vol_mask = vol < cfg['strategy']['min_vol']
    signals.loc[low_vol_mask, ['long','short','exit']] = 0

    # Filter: force exit after max holding period
    max_h = cfg['strategy']['max_holding']
    hold_days = 0
    for date, row in signals.iterrows():
        if row['long'] == 1 or row['short'] == 1:
            hold_days = 1
        elif hold_days > 0:
            hold_days += 1
            if hold_days > max_h:
                signals.at[date, 'exit'] = 1
                hold_days = 0
        else:
            hold_days = 0

    # Build position vector
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

    # Run backtest
    returns, cum_returns = backtest(
        prices, signals, beta,
        tc=cfg['backtest']['tc_per_trade']
    )
    report_performance(returns, cum_returns)

if __name__ == '__main__':
    main()
