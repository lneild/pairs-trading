Notes:

Paris Trading: pick 2 stocks, or assets, whose prices move in parallel. Instead of betting that the market will go up or down, you bet that the gap between two stocks will be narrow. If there is a gap, you either sell the higher one (it might go down towards the other) or buy the lower one (it might go up to the higher one)


config.yml: sets params for the fcn, tweak a little and try different data, time frame, etc 

Look for pairs automatically: instead of hand picking 2 stocks that might move together, we scan the whole universe (ex: FAANG stocks) to find best to picl 
* uses a statistical test called cointegration to spot pairs who prices share a stable long-term relationship 

Data: Imported from `yfinance` python package which pulls from Yahoo Fiannce  

`yf.download(['MSFT','AAPL'], start='2020-01-01', end='2025-06-01'` will return adjusted close prices of rMicrosoft and Apple in that date range 

Interpretting Output:
`poetry run python data_fetch.py`
Gives you the adj closing price in USD for the ticker on that day (showing the last 5 rows of data)

`poetry run python strat.py`
Each row shows one trading day (gaps for holidays and weekends). Signal Columns (`long`, `short`, `exit`)
* `long` : a value of 1 means your strategy is signalling to go long the spread - you buy the first asste (y) and simultaneously short the second asset (x) in proportion to the hedge ratio
* `short` : A value of 1 means the opposite entry. you short the spread by selling y and buying x.
* `exit` :  A value of 1 means close any existing position (flat both legs) because the spead has reverted enough to its mean

Hedge Ratio: This number comes from the regression (𝑦 ≈ 𝛽𝑥). In practice it tells you how to size your positions so the two legs offset each other’s market exposure. For example, if you get a long signal:

* Buy 1 share of y (the first ticker in your pair), and
* Short 1.4865 shares of x (the second ticker).
