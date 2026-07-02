# Cross-sectional Rough Volatility Alpha

This copy modifies the original project in two ways:

1. Backtest summaries now report `monthly_return` instead of `ann_return`.
2. `src/visualize_monthly_vs_benchmark.py` creates comparison plots against a local broad-market benchmark proxy.

## Universe

The current universe is not restricted to S&P 500 constituents.

`src/universe.py` builds the stock pool from CRSP daily stocks and keeps NYSE/AMEX/NASDAQ listings (`exchcd` in 1, 2, 3), then applies filters:

- price >= 5
- 20-day average dollar volume >= 1,000,000
- listed for at least 252 days
- not delisted as of the rebalance date

So the universe is closer to a liquid US listed equity universe, not SPY/S&P 500 only.

## Monthly Return

`src/backtest.py` now computes:

```text
monthly_return = geometric mean of rebalance-period net returns
```

Sharpe and Calmar are still kept on the usual annualized scale:

```text
sharpe = monthly_return * 12 / annualized_volatility
calmar = monthly_return * 12 / abs(max_drawdown)
```

## Benchmark

No exact S&P 500 or SPY return file is included in the folder. The visualization script uses the local Fama-French daily market return:

```text
market_proxy = Mkt-RF + RF
```

This is a broad US equity market proxy, not the exact S&P 500 index.

## Raw CSV Location

`src/data_loader.py` first looks for raw WRDS CSV files in:

```text
data/raw/downloads/
```

If they are not there, it automatically falls back to:

```text
C:\Users\johnh\Finance\finance project\ML
```

This avoids copying the large CRSP files into the project folder.

Run Window 8 monthly backtest. Window 8 is the 2023 out-of-sample test period:

```bash
python src/monthly_backtest.py
```

Run per-model monthly return plots:

```bash
python src/visualize_monthly_vs_benchmark.py
```

Run Window 8 daily holding-period backtest and daily cumulative plots:

```bash
python src/daily_backtest.py
python src/visualize_daily_vs_benchmark.py
```

Outputs:

- `data/reports/portfolio_monthly_returns_window8.csv`
- `data/reports/monthly_backtest_summary_window8.csv`
- `data/figures/monthly_returns_window8_ridge_long_short_fees_vs_market.png`
- `data/figures/monthly_returns_window8_lgbm_long_short_fees_vs_market.png`
- `data/figures/monthly_returns_window8_ranker_long_short_fees_vs_market.png`
- `data/reports/portfolio_daily_returns_window8.csv`
- `data/figures/daily_cumulative_returns_window8_ridge_long_short_fees_vs_market.png`
- `data/figures/daily_cumulative_returns_window8_lgbm_long_short_fees_vs_market.png`
- `data/figures/daily_cumulative_returns_window8_ranker_long_short_fees_vs_market.png`

Each figure contains one ML model, several transaction-fee lines, and the
market benchmark proxy line. The plots require `data/processed/predictions.parquet`
and `data/processed/labels.parquet`; if `portfolio_monthly_returns_window8.csv` is not
present, the plotting script will automatically run `src/monthly_backtest.py`.

For daily charts, signals remain monthly. The daily backtest expands each
monthly portfolio into the next 20 CRSP trading days and computes daily
portfolio returns while charging transaction costs on each rebalance entry day.
