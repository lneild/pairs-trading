import yaml
from data_fetch import fetch_prices
from pair_selection import fetch_universe, find_pairs
from strat import generate_signals
from backtest import backtest
from performance import report_performance, sharpe, max_drawdown
import pandas as pd
from itertools import product

def load_cfg(path='config.yml'):
    return yaml.safe_load(open(path))

def test_timeframes(tickers, start, end, interval, cfg):
    from pandas import to_datetime
    start, end = to_datetime(start), to_datetime(end)
    window = pd.Timedelta(days=365)
    results = []
    lookbacks = [10, 20, 30]
    z_enters = [1.5, 2.0, 2.5]
    z_exits = [0.5, 1.0]
    max_holdings = [5, 10, 20]
    
    current_start = start
    while current_start + window <= end:
        current_end = current_start + window
        prices = fetch_prices(tickers, current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d'), interval)
        if prices.empty or prices.isna().all().any():
            current_start += pd.Timedelta(days=30)
            continue
        for lb, ze, zx, mh in product(lookbacks, z_enters, z_exits, max_holdings):
            signals, beta = generate_signals(prices, lookback=lb, z_enter=ze, z_exit=zx)
            if signals.empty:
                continue
            spread = prices.iloc[:, 0] - beta * prices.iloc[:, 1]
            vol = spread.rolling(lb).std()
            # Replace NaN or zero volatility with a small positive value to avoid division issues
            vol = vol.fillna(0.0001).replace(0, 0.0001)
            low_vol_mask = vol < cfg['strategy']['min_vol']
            signals.loc[low_vol_mask, ['long', 'short', 'exit']] = 0
            hold_days = 0
            for date, row in signals.iterrows():
                if row['long'] == 1 or row['short'] == 1:
                    hold_days = 1
                elif hold_days > 0:
                    hold_days += 1
                    if hold_days > mh:
                        signals.at[date, 'exit'] = 1
                        hold_days = 0
                else:
                    hold_days = 0
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
            if not returns.empty and not cum_returns.empty:
                total_return = cum_returns.iloc[-1]
                results.append({
                    'start': current_start,
                    'end': current_end,
                    'lookback': lb,
                    'z_enter': ze,
                    'z_exit': zx,
                    'max_holding': mh,
                    'total_return': total_return,
                    'sharpe': sharpe(returns),
                    'max_dd': max_drawdown(cum_returns)
                })
        current_start += pd.Timedelta(days=30)
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('total_return', ascending=False)
    return results_df

def main():
    cfg = load_cfg()
    try:
        uni = fetch_universe(cfg['tickers']['universe'],
                            cfg['data']['start'],
                            cfg['data']['end'])
        if uni.empty:
            print("Error: No data fetched for universe.")
            return
        p_thresh = cfg['pair_selection']['p_thresh']
        corr_thresh = cfg['pair_selection']['corr_thresh']
        vol_thresh = cfg['pair_selection']['vol_thresh']
        pairs, _ = find_pairs(uni, p_thresh, corr_thresh, vol_thresh)
        if not pairs:
            print("Warning: No cointegrated pairs found. Using default pair.")
            t1, t2 = cfg['tickers']['pair']
        else:
            top_pairs = pairs[:3] if len(pairs) >= 3 else pairs
            all_results = []
            for t1, t2 in top_pairs:
                results = test_timeframes(
                    [t1, t2],
                    cfg['data']['start'],
                    cfg['data']['end'],
                    cfg['data']['interval'],
                    cfg
                )
                if not results.empty:
                    results['pair'] = f"{t1},{t2}"
                    all_results.append(results)
            if not all_results:
                print("Error: No valid results from timeframes.")
                return
            all_results = pd.concat(all_results, ignore_index=True)
            all_results = all_results.sort_values('total_return', ascending=False)
            # Write results to file instead of printing to terminal
            with open('results.txt', 'w') as f:
                f.write("\nTimeframe & Parameter Performance (sorted by total return):\n")
                f.write(all_results.to_markdown())
                best = all_results.iloc[0]
                f.write(f"\n\nBest Timeframe: {best['start']} to {best['end']}\n")
                f.write(f"Best Pair: {best['pair']}\n")
                f.write(f"Best Parameters: lookback={best['lookback']}, z_enter={best['z_enter']}, "
                        f"z_exit={best['z_exit']}, max_holding={best['max_holding']}\n")
                f.write(f"Total Return: {best['total_return']:.2f}\n")
            print("Results have been saved to 'results.txt'. Please check the file for details.")
            t1, t2 = best['pair'].split(',')
        prices = fetch_prices([t1, t2], best['start'].strftime('%Y-%m-%d'), best['end'].strftime('%Y-%m-%d'), cfg['data']['interval'])
        if prices.empty or prices.isna().all().any():
            print("Error: No valid price data for backtest.")
            return
        signals, beta = generate_signals(prices, lookback=best['lookback'], z_enter=best['z_enter'], z_exit=best['z_exit'])
        spread = prices.iloc[:, 0] - beta * prices.iloc[:, 1]
        vol = spread.rolling(best['lookback']).std()
        vol = vol.fillna(0.0001).replace(0, 0.0001)
        low_vol_mask = vol < cfg['strategy']['min_vol']
        signals.loc[low_vol_mask, ['long', 'short', 'exit']] = 0
        max_h = best['max_holding']
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
        if not returns.empty and not cum_returns.empty:
            report_performance(returns, cum_returns, signals)
        else:
            print("Error: Backtest returned empty results.")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == '__main__':
    main()