"""
Plot monthly strategy returns by ML model and transaction fee.

For each ML model, this script creates one chart containing:
  - one monthly-return line for each transaction cost
  - one broad US market benchmark line

The benchmark uses the local Fama-French daily market return, Mkt-RF + RF,
because this folder does not currently include an exact SPY/S&P 500 return file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_PROCESSED
from monthly_backtest import run_window8_monthly_backtest


ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = ROOT.parent
REPORTS = ROOT / "data" / "reports"
FIGURES = ROOT / "data" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def _read_ff_csv(path: Path) -> pd.DataFrame:
    ff = pd.read_csv(path, skiprows=4)
    ff = ff.rename(columns={ff.columns[0]: "date"})
    ff = ff[pd.to_numeric(ff["date"], errors="coerce").notna()].copy()
    ff["date"] = pd.to_datetime(ff["date"].astype(int).astype(str), format="%Y%m%d")
    ff["market_return"] = (
        pd.to_numeric(ff["Mkt-RF"], errors="coerce")
        + pd.to_numeric(ff["RF"], errors="coerce")
    ) / 100
    return ff[["date", "market_return"]].dropna()


def load_market_proxy(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    candidates = [
        ML_ROOT / "ff_factors_daily.csv",
        ROOT / "data" / "raw" / "downloads" / "ff_factors_daily.csv",
        ROOT / "data" / "raw" / "ff_factors_daily.csv",
    ]
    ff_path = next((p for p in candidates if p.exists()), None)
    if ff_path is None:
        raise FileNotFoundError("Could not find ff_factors_daily.csv for market benchmark.")

    ff = _read_ff_csv(ff_path)
    ff = ff[(ff["date"] >= start) & (ff["date"] <= end)]
    monthly = (1 + ff.set_index("date")["market_return"]).resample("ME").prod() - 1
    monthly.name = "market_proxy"
    return monthly.dropna()


def load_or_build_monthly_returns(window: str = "window8") -> pd.DataFrame:
    path = REPORTS / f"portfolio_monthly_returns_{window}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        predictions_path = DATA_PROCESSED / "predictions.parquet"
        labels_path = DATA_PROCESSED / "labels.parquet"
        if not predictions_path.exists() or not labels_path.exists():
            missing = [str(p) for p in [predictions_path, labels_path] if not p.exists()]
            raise FileNotFoundError(
                "Cannot draw monthly return time-series yet. Missing processed files: "
                f"{missing}. Run the data/feature/training pipeline first."
            )
        if window != "window8":
            raise ValueError(f"Unsupported automatic build window: {window}")
        df, _ = run_window8_monthly_backtest()

    required = {"date", "model", "cost_bps", "portfolio", "net_return"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"portfolio_monthly_returns.csv is missing columns: {sorted(missing_cols)}")

    df["date"] = pd.to_datetime(df["date"])
    df["cost_bps"] = df["cost_bps"].astype(int)
    return df.sort_values(["model", "portfolio", "cost_bps", "date"]).reset_index(drop=True)


def plot_model_fee_lines(
    monthly_returns: pd.DataFrame,
    market_monthly: pd.Series,
    model: str,
    portfolio: str = "long_short",
    window: str = "window8",
) -> Path:
    data = monthly_returns[
        (monthly_returns["model"] == model)
        & (monthly_returns["portfolio"] == portfolio)
    ].copy()
    if data.empty:
        raise ValueError(f"No monthly returns found for model={model}, portfolio={portfolio}")

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    cmap = plt.get_cmap("viridis")
    fees = sorted(data["cost_bps"].unique())

    for idx, fee in enumerate(fees):
        g = data[data["cost_bps"] == fee].sort_values("date")
        color = cmap(idx / max(1, len(fees) - 1))
        ax.plot(
            g["date"],
            g["net_return"],
            linewidth=1.8,
            color=color,
            label=f"{fee} bps",
        )

    start, end = data["date"].min(), data["date"].max()
    mkt = market_monthly[(market_monthly.index >= start) & (market_monthly.index <= end)]
    ax.plot(
        mkt.index,
        mkt.values,
        color="#111827",
        linewidth=2.4,
        linestyle="--",
        label="Market proxy",
    )

    ax.axhline(0, color="#64748b", linewidth=0.9)
    ax.set_title(f"{model}: Window 8 test monthly {portfolio} returns by transaction fee vs market")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly return")
    ax.grid(alpha=0.25)
    ax.legend(title="Transaction fee", ncol=3)
    fig.tight_layout()

    out = FIGURES / f"monthly_returns_{window}_{model}_{portfolio}_fees_vs_market.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_all_models(portfolio: str = "long_short", window: str = "window8") -> list[Path]:
    monthly_returns = load_or_build_monthly_returns(window=window)
    start = monthly_returns["date"].min()
    end = monthly_returns["date"].max()
    market_monthly = load_market_proxy(start, end)

    outputs = []
    for model in sorted(monthly_returns["model"].unique()):
        outputs.append(plot_model_fee_lines(monthly_returns, market_monthly, model, portfolio, window=window))

    return outputs


def main() -> None:
    try:
        outputs = plot_all_models(portfolio="long_short", window="window8")
    except FileNotFoundError as exc:
        print(exc)
        return

    print("Saved monthly return charts:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
