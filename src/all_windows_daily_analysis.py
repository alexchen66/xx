"""
Run and visualize all walk-forward daily backtests for 2016-2023.

The backtest uses monthly rebalance signals and daily holding-period returns.
Portfolio weights default to equal weights, while portfolio.py keeps the
weighting interface available for alternatives such as rank weighting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from daily_backtest import run_all_windows_daily_backtest
from config import DATA_PROCESSED
from visualize_daily_vs_benchmark import load_market_proxy


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
FIGURES = ROOT / "data" / "figures" / "daily_all_windows"
YEARLY_FIGURES = FIGURES / "by_year"
FULL_FIGURES = FIGURES / "full_period"


def load_or_run_all_windows(force: bool = True) -> pd.DataFrame:
    path = REPORTS / "portfolio_daily_returns_all_windows.csv"
    if force or not path.exists():
        panel = run_all_windows_daily_backtest()
    else:
        panel = pd.read_csv(path, parse_dates=["date", "hold_date"])

    panel["date"] = pd.to_datetime(panel["date"])
    panel["hold_date"] = pd.to_datetime(panel["hold_date"])
    panel["year"] = panel["hold_date"].dt.year
    panel["cost_bps"] = panel["cost_bps"].astype(int)
    return panel.sort_values(["model", "portfolio", "cost_bps", "hold_date"]).reset_index(drop=True)


def _daily_series(data: pd.DataFrame, model: str, portfolio: str, cost_bps: int) -> pd.Series:
    g = data[
        (data["model"] == model)
        & (data["portfolio"] == portfolio)
        & (data["cost_bps"] == cost_bps)
    ]
    return g.groupby("hold_date")["net_return"].sum().sort_index()


def _wealth(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def _max_drawdown(returns: pd.Series) -> float:
    wealth = _wealth(returns)
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def _sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return float(np.sqrt(252) * returns.mean() / std)


def _plot_curves(
    data: pd.DataFrame,
    market_daily: pd.Series,
    model: str,
    portfolio: str,
    out: Path,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    cmap = plt.get_cmap("viridis")
    fees = sorted(data["cost_bps"].unique())

    for idx, fee in enumerate(fees):
        daily = _daily_series(data, model, portfolio, fee)
        if daily.empty:
            continue
        color = cmap(idx / max(1, len(fees) - 1))
        ax.plot(_wealth(daily).index, _wealth(daily).values, linewidth=1.15, color=color, label=f"{fee} bps")

    start, end = data["hold_date"].min(), data["hold_date"].max()
    mkt = market_daily[(market_daily.index >= start) & (market_daily.index <= end)]
    ax.plot(
        _wealth(mkt).index,
        _wealth(mkt).values,
        color="#111827",
        linewidth=1.6,
        linestyle="--",
        label="Market proxy",
    )

    ax.axhline(1.0, color="#64748b", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend(title="Transaction fee", ncol=3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_yearly(data: pd.DataFrame, market_daily: pd.Series) -> list[Path]:
    outputs = []
    for year in sorted(data["year"].unique()):
        year_data = data[data["year"] == year].copy()
        for model in sorted(year_data["model"].unique()):
            for portfolio in ["long_only", "long_short"]:
                out = YEARLY_FIGURES / f"daily_cumulative_returns_{year}_{model}_{portfolio}_fees_vs_market.png"
                title = f"{model}: {year} daily cumulative {portfolio} returns by fee vs market"
                outputs.append(_plot_curves(year_data, market_daily, model, portfolio, out, title))
    return outputs


def plot_full_period(data: pd.DataFrame, market_daily: pd.Series) -> list[Path]:
    outputs = []
    for model in sorted(data["model"].unique()):
        for portfolio in ["long_only", "long_short"]:
            out = FULL_FIGURES / f"daily_cumulative_returns_2016_2023_{model}_{portfolio}_fees_vs_market.png"
            title = f"{model}: 2016-2023 daily cumulative {portfolio} returns by fee vs market"
            outputs.append(_plot_curves(data, market_daily, model, portfolio, out, title))
    return outputs


def make_backtest_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = data.groupby(["year", "model", "portfolio", "cost_bps"], sort=True)
    for (year, model, portfolio, cost_bps), g in grouped:
        daily = g.groupby("hold_date")["net_return"].sum().sort_index()
        entry_turnover = g.groupby("date")["turnover"].first()
        rows.append(
            {
                "year": year,
                "model": model,
                "portfolio": portfolio,
                "cost_bps": cost_bps,
                "annual_return": float(_wealth(daily).iloc[-1] - 1.0) if not daily.empty else np.nan,
                "annualized_vol": float(np.sqrt(252) * daily.std(ddof=1)) if len(daily) > 1 else np.nan,
                "sharpe": _sharpe(daily),
                "max_drawdown": _max_drawdown(daily),
                "mean_daily_return": float(daily.mean()) if not daily.empty else np.nan,
                "trading_days": int(daily.shape[0]),
                "avg_rebalance_turnover": float(entry_turnover.mean()) if not entry_turnover.empty else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    out = REPORTS / "daily_backtest_summary_all_windows_by_year.csv"
    summary.to_csv(out, index=False)
    return summary


def make_ic_summary() -> pd.DataFrame:
    predictions = pd.read_parquet(DATA_PROCESSED / "predictions.parquet")
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions = predictions[(predictions["date"] >= "2016-01-01") & (predictions["date"] <= "2023-12-31")]

    score_cols = [c for c in predictions.columns if c.startswith("score_")]
    rows = []
    for date, g in predictions.groupby("date"):
        for score_col in score_cols:
            model = score_col.replace("score_", "")
            pearson_ic = g[score_col].corr(g["y_xs"], method="pearson")
            spearman_ic = g[score_col].corr(g["y_xs"], method="spearman")
            rows.append(
                {
                    "date": date,
                    "year": date.year,
                    "model": model,
                    "pearson_ic": pearson_ic,
                    "spearman_rank_ic": spearman_ic,
                    "n": int(g[[score_col, "y_xs"]].dropna().shape[0]),
                }
            )

    daily_ic = pd.DataFrame(rows)
    daily_ic.to_csv(REPORTS / "ic_by_rebalance_date_all_windows.csv", index=False)

    summary_rows = []
    for (year, model), g in daily_ic.groupby(["year", "model"], sort=True):
        for col in ["pearson_ic", "spearman_rank_ic"]:
            vals = g[col].dropna()
            std = vals.std(ddof=1)
            summary_rows.append(
                {
                    "year": year,
                    "model": model,
                    "ic_type": col,
                    "mean_ic": float(vals.mean()) if not vals.empty else np.nan,
                    "median_ic": float(vals.median()) if not vals.empty else np.nan,
                    "std_ic": float(std) if not vals.empty else np.nan,
                    "ic_ir": float(vals.mean() / std) if len(vals) > 1 and std != 0 else np.nan,
                    "positive_ic_rate": float((vals > 0).mean()) if not vals.empty else np.nan,
                    "n_rebalance_dates": int(vals.shape[0]),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORTS / "ic_summary_all_windows_by_year.csv", index=False)
    return summary


def main() -> None:
    data = load_or_run_all_windows(force=False)
    start, end = data["hold_date"].min(), data["hold_date"].max()
    market_daily = load_market_proxy(start, end)

    yearly_outputs = plot_yearly(data, market_daily)
    full_outputs = plot_full_period(data, market_daily)
    summary = make_backtest_summary(data)
    ic_summary = make_ic_summary()

    print(f"Saved all-windows daily returns: {REPORTS / 'portfolio_daily_returns_all_windows.csv'}")
    print(f"Saved {len(yearly_outputs)} yearly charts under: {YEARLY_FIGURES}")
    print(f"Saved {len(full_outputs)} full-period charts under: {FULL_FIGURES}")
    print(f"Saved backtest summary rows: {len(summary)}")
    print(f"Saved IC summary rows: {len(ic_summary)}")


if __name__ == "__main__":
    main()
