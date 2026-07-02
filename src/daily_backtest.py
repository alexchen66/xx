"""
Daily holding-period backtest from monthly model signals.

Signals are still monthly. For each rebalance signal date, the script builds
monthly portfolios from model scores, then expands that holding across CRSP
daily returns. Transaction costs are charged once on the first holding day of
each rebalance period. By default, each monthly signal is held until the next
monthly rebalance entry day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import COST_BPS_GRID, DATA_PROCESSED, HOLDING_PERIOD
from portfolio import build_long_only, build_long_short, compute_turnover


REPORTS = DATA_PROCESSED.parent / "reports"
ALL_WINDOWS_START = "2016-01-01"
ALL_WINDOWS_END = "2023-12-31"
WINDOW8_TEST_START = "2023-01-01"
WINDOW8_TEST_END = "2023-12-31"


def _filter_predictions(
    predictions: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"])
    if start_date is not None:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out["date"] <= pd.Timestamp(end_date)]
    if out.empty:
        raise ValueError(f"No predictions after filtering start={start_date}, end={end_date}")
    return out


def _make_holding_calendar(
    signal_dates: list[pd.Timestamp],
    trading_days: np.ndarray,
    hold_to_next_rebalance: bool = True,
) -> pd.DataFrame:
    records = []
    trading_days = pd.DatetimeIndex(trading_days)
    signal_dates = [pd.Timestamp(d) for d in sorted(signal_dates)]
    entry_dates = {}
    for signal_date in signal_dates:
        idx = trading_days.searchsorted(signal_date)
        entry_dates[signal_date] = trading_days[idx + 1] if idx + 1 < len(trading_days) else pd.NaT

    for i, signal_date in enumerate(signal_dates):
        entry_date = entry_dates[signal_date]
        if pd.isna(entry_date):
            continue

        entry_idx = trading_days.searchsorted(entry_date)
        if hold_to_next_rebalance and i + 1 < len(signal_dates):
            next_entry = entry_dates[signal_dates[i + 1]]
            exit_idx = trading_days.searchsorted(next_entry) if not pd.isna(next_entry) else len(trading_days)
        else:
            exit_idx = min(entry_idx + HOLDING_PERIOD, len(trading_days))

        for hold_date in trading_days[entry_idx:exit_idx]:
            records.append({"date": signal_date, "hold_date": hold_date})
    return pd.DataFrame(records)


def _portfolio_daily_returns(
    portfolio: pd.DataFrame,
    crsp_returns: pd.DataFrame,
    holding_calendar: pd.DataFrame,
    cost_bps: int,
) -> pd.DataFrame:
    turnover = compute_turnover(portfolio).rename("turnover").reset_index()
    expanded = portfolio.merge(holding_calendar, on="date", how="inner")

    merged = expanded.merge(
        crsp_returns,
        left_on=["hold_date", "permno"],
        right_on=["date", "permno"],
        how="left",
        suffixes=("", "_ret"),
    )
    merged["ret"] = merged["ret"].fillna(0.0)
    merged["contribution"] = merged["weight"] * merged["ret"]

    daily = (
        merged.groupby(["date", "hold_date"])["contribution"]
        .sum()
        .rename("gross_return")
        .reset_index()
    )
    daily = daily.merge(turnover, on="date", how="left")
    daily["is_entry_day"] = daily.groupby("date")["hold_date"].transform("min").eq(daily["hold_date"])
    daily["cost"] = np.where(daily["is_entry_day"], daily["turnover"] * (cost_bps / 10_000), 0.0)
    daily["net_return"] = daily["gross_return"] - daily["cost"]
    return daily


def build_daily_return_panel(
    predictions: pd.DataFrame,
    crsp: pd.DataFrame,
    cost_bps_grid: list[int] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    hold_to_next_rebalance: bool = True,
    weighting: str = "equal",
) -> pd.DataFrame:
    if cost_bps_grid is None:
        cost_bps_grid = COST_BPS_GRID

    predictions = _filter_predictions(predictions, start_date, end_date)
    crsp_returns = crsp[["date", "permno", "ret"]].copy()
    crsp_returns["date"] = pd.to_datetime(crsp_returns["date"])
    crsp_returns["ret"] = crsp_returns["ret"].fillna(0.0)

    trading_days = np.sort(crsp_returns["date"].unique())
    signal_dates = sorted(predictions["date"].unique())
    holding_calendar = _make_holding_calendar(
        signal_dates,
        trading_days,
        hold_to_next_rebalance=hold_to_next_rebalance,
    )

    score_cols = [c for c in predictions.columns if c.startswith("score_")]
    if not score_cols:
        raise ValueError("predictions.parquet has no score_* columns.")

    rows = []
    for score_col in score_cols:
        model = score_col.replace("score_", "")
        portfolios = {
            "long_only": build_long_only(predictions, score_col, weighting=weighting),
            "long_short": build_long_short(predictions, score_col, weighting=weighting),
        }
        for portfolio_name, portfolio in portfolios.items():
            for cost_bps in cost_bps_grid:
                daily = _portfolio_daily_returns(portfolio, crsp_returns, holding_calendar, cost_bps)
                daily["model"] = model
                daily["cost_bps"] = cost_bps
                daily["portfolio"] = portfolio_name
                rows.append(
                    daily[
                        [
                            "date",
                            "hold_date",
                            "model",
                            "cost_bps",
                            "portfolio",
                            "gross_return",
                            "net_return",
                            "turnover",
                            "cost",
                        ]
                    ]
                )

    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["model", "portfolio", "cost_bps", "hold_date"]).reset_index(drop=True)


def run_daily_backtest(
    start_date: str | None = None,
    end_date: str | None = None,
    output_suffix: str = "",
    hold_to_next_rebalance: bool = True,
    weighting: str = "equal",
) -> pd.DataFrame:
    predictions_path = DATA_PROCESSED / "predictions.parquet"
    crsp_path = DATA_PROCESSED / "crsp_clean.parquet"
    if not predictions_path.exists() or not crsp_path.exists():
        missing = [str(p) for p in [predictions_path, crsp_path] if not p.exists()]
        raise FileNotFoundError(f"Daily backtest requires missing files: {missing}")

    predictions = pd.read_parquet(predictions_path)
    crsp = pd.read_parquet(crsp_path, columns=["date", "permno", "ret"])
    panel = build_daily_return_panel(
        predictions,
        crsp,
        start_date=start_date,
        end_date=end_date,
        hold_to_next_rebalance=hold_to_next_rebalance,
        weighting=weighting,
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""
    out = REPORTS / f"portfolio_daily_returns{suffix}.csv"
    panel.to_csv(out, index=False)
    return panel


def run_window8_daily_backtest() -> pd.DataFrame:
    return run_daily_backtest(
        start_date=WINDOW8_TEST_START,
        end_date=WINDOW8_TEST_END,
        output_suffix="window8",
        hold_to_next_rebalance=True,
        weighting="equal",
    )


def run_all_windows_daily_backtest() -> pd.DataFrame:
    return run_daily_backtest(
        start_date=ALL_WINDOWS_START,
        end_date=ALL_WINDOWS_END,
        output_suffix="all_windows",
        hold_to_next_rebalance=True,
        weighting="equal",
    )


def main() -> None:
    panel = run_window8_daily_backtest()
    print(f"Saved {REPORTS / 'portfolio_daily_returns_window8.csv'} ({len(panel):,} rows)")


if __name__ == "__main__":
    main()
