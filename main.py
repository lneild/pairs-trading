import yaml
from data_fetch import fetch_prices
from pair_selection import fetch_universe, find_pairs
from strategy import generate_signals
from backtest import backtest
from performance import report_performance


def load_cfg(path='config.yml'):
    return yaml.safe_load(open(path))


def main():
    cfg = load_cfg()

    # 1. find best pair
    uni    = fetch_universe(cfg['tickers']['universe'], cfg['data']['start'], cfg['data']['end'])
    p_thresh = cfg['pair_selection']['p_thresh']
    pairs, pvals = find_pairs(uni, p_thresh=p_thresh)
    t1, t2 = pairs[0] if pairs else cfg['tickers']['pair']

    # 2. fetch data for chosen pair
    prices = fetch_prices([t1, t2], cfg['data']['start'], cfg['data']['end'], cfg['data']['interval'])

    # 3. generate signals
    signals, beta = generate_signals(prices,
                                     lookback=cfg['strategy']['lookback'],
                                     z_enter=cfg['strategy']['z_enter'],
                                     z_exit=cfg['strategy']['z_exit'])

    # 4. build position
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

    # 5. backtest & report
    returns, cum_returns = backtest(prices, signals, beta, tc=cfg['backtest']['tc_per_trade'])
    report_performance(returns, cum_returns)

if __name__ == '__main__':
    main()
