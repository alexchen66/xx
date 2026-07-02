"""
Monthly portfolio backtest from model predictions.

Inputs
------
data/processed/predictions.parquet
    Must contain date, permno, and score_* columns, for example
    score_ridge, score_lgbm, score_ranker.

data/processed/labels.parquet
    Must contain date, permno, and r_fwd. r_fwd is the realized forward
    rebalance-period return used as the next-month return.

Outputs
-------
data/reports/portfolio_monthly_returns.csv
    One row per date/model/transaction-cost/portfolio.

data/reports/monthly_backtest_summary.csv
    Monthly-return summary by model, transaction cost, and portfolio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import COST_BPS_GRID, DATA_PROCESSED
from portfolio import build_long_only, build_long_short, compute_turnover


REPORTS = DATA_PROCESSED.parent / "reports"
WINDOW8_TEST_START = "2023-01-01"
WINDOW8_TEST_END = "2023-12-31"


def compute_portfolio_returns(
    portfolio: pd.DataFrame,
    realized_returns: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    """Compute gross/net monthly returns for a dated portfolio panel."""
    turnover = compute_turnover(portfolio)

    merged = portfolio.merge(realized_returns, on=["date", "permno"], how="left")
    merged["r_fwd"] = merged["r_fwd"].fillna(0.0)
    merged["contribution"] = merged["weight"] * merged["r_fwd"]

    out = merged.groupby("date")["contribution"].sum().rename("gross_return").to_frame()
    out["turnover"] = turnover
    out["net_return"] = out["gross_return"] - out["turnover"] * (cost_bps / 10_000)
    return out.reset_index()


def summarize_monthly_returns(returns: pd.Series) -> dict:
    """Summarize a monthly return series without annualizing the return field."""
    r = returns.dropna()
    monthly_return = (1 + r).prod() ** (1 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = (monthly_return * 12) / ann_vol if ann_vol > 0 else np.nan

    wealth = (1 + r).cumprod()
    drawdown = (wealth - wealth.cummax()) / wealth.cummax()
    max_drawdown = drawdown.min()
    calmar = (monthly_return * 12) / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return {
        "monthly_return": monthly_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "hit_ratio": (r > 0).mean(),
        "n_periods": len(r),
    }


def build_monthly_return_panel(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    cost_bps_grid: list[int] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Build monthly returns for every model, fee, and portfolio type."""
    if cost_bps_grid is None:
        cost_bps_grid = COST_BPS_GRID

    predictions = predictions.copy()
    labels = labels.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    if start_date is not None:
        start = pd.Timestamp(start_date)
        predictions = predictions[predictions["date"] >= start]
        labels = labels[labels["date"] >= start]
    if end_date is not None:
        end = pd.Timestamp(end_date)
        predictions = predictions[predictions["date"] <= end]
        labels = labels[labels["date"] <= end]

    if predictions.empty:
        raise ValueError(
            f"No predictions remain after date filtering: start={start_date}, end={end_date}"
        )

    score_cols = [c for c in predictions.columns if c.startswith("score_")]
    if not score_cols:
        raise ValueError("predictions.parquet has no score_* columns.")

    realized = labels[["date", "permno", "r_fwd"]].copy()
    rows = []

    for score_col in score_cols:
        model = score_col.replace("score_", "")
        portfolio_builders = {
            "long_only": build_long_only(predictions, score_col),
            "long_short": build_long_short(predictions, score_col),
        }

        for portfolio_name, portfolio in portfolio_builders.items():
            for cost_bps in cost_bps_grid:
                ret = compute_portfolio_returns(portfolio, realized, cost_bps)
                ret["model"] = model
                ret["cost_bps"] = cost_bps
                ret["portfolio"] = portfolio_name
                rows.append(
                    ret[
                        [
                            "date",
                            "model",
                            "cost_bps",
                            "portfolio",
                            "gross_return",
                            "net_return",
                            "turnover",
                        ]
                    ]
                )

    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["model", "portfolio", "cost_bps", "date"]).reset_index(drop=True)


def summarize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cost_bps, portfolio), g in panel.groupby(["model", "cost_bps", "portfolio"]):
        summary = summarize_monthly_returns(g.sort_values("date")["net_return"])
        rows.append(
            {
                "model": model,
                "cost_bps": cost_bps,
                "portfolio": portfolio,
                **summary,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "portfolio", "cost_bps"]).reset_index(drop=True)


def run_monthly_backtest(
    start_date: str | None = None,
    end_date: str | None = None,
    output_suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_path = DATA_PROCESSED / "predictions.parquet"
    labels_path = DATA_PROCESSED / "labels.parquet"
    if not predictions_path.exists() or not labels_path.exists():
        missing = [str(p) for p in [predictions_path, labels_path] if not p.exists()]
        raise FileNotFoundError(
            "Monthly backtest requires processed prediction and label files. "
            f"Missing: {missing}"
        )

    predictions = pd.read_parquet(predictions_path)
    labels = pd.read_parquet(labels_path)
    panel = build_monthly_return_panel(
        predictions,
        labels,
        start_date=start_date,
        end_date=end_date,
    )
    summary = summarize_panel(panel)

    REPORTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""
    panel.to_csv(REPORTS / f"portfolio_monthly_returns{suffix}.csv", index=False)
    summary.to_csv(REPORTS / f"monthly_backtest_summary{suffix}.csv", index=False)
    return panel, summary


def run_window8_monthly_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the monthly backtest only on Window 8 test dates: calendar year 2023."""
    return run_monthly_backtest(
        start_date=WINDOW8_TEST_START,
        end_date=WINDOW8_TEST_END,
        output_suffix="window8",
    )


def main() -> None:
    panel, summary = run_window8_monthly_backtest()
    print(f"Saved {REPORTS / 'portfolio_monthly_returns_window8.csv'} ({len(panel):,} rows)")
    print(f"Saved {REPORTS / 'monthly_backtest_summary_window8.csv'} ({len(summary):,} rows)")


if __name__ == "__main__":
    main()
