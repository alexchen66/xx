"""
Plot daily cumulative returns from monthly-rebalanced portfolios.

Each output chart contains one ML model, multiple transaction-fee curves, and
a broad US market benchmark proxy based on Fama-French Mkt-RF + RF.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from daily_backtest import run_window8_daily_backtest


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
    return ff.set_index("date")["market_return"].sort_index()


def load_or_build_daily_returns(window: str = "window8") -> pd.DataFrame:
    path = REPORTS / f"portfolio_daily_returns_{window}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date", "hold_date"])
    else:
        if window != "window8":
            raise ValueError(f"Unsupported automatic build window: {window}")
        df = run_window8_daily_backtest()

    required = {"hold_date", "model", "cost_bps", "portfolio", "net_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["hold_date"] = pd.to_datetime(df["hold_date"])
    df["cost_bps"] = df["cost_bps"].astype(int)
    return df.sort_values(["model", "portfolio", "cost_bps", "hold_date"]).reset_index(drop=True)


def plot_model_daily_curves(
    daily_returns: pd.DataFrame,
    market_daily: pd.Series,
    model: str,
    portfolio: str = "long_short",
    window: str = "window8",
) -> Path:
    data = daily_returns[
        (daily_returns["model"] == model)
        & (daily_returns["portfolio"] == portfolio)
    ].copy()
    if data.empty:
        raise ValueError(f"No daily returns for model={model}, portfolio={portfolio}")

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    cmap = plt.get_cmap("viridis")
    fees = sorted(data["cost_bps"].unique())

    for idx, fee in enumerate(fees):
        g = data[data["cost_bps"] == fee].sort_values("hold_date")
        daily = g.groupby("hold_date")["net_return"].sum().sort_index()
        wealth = (1 + daily).cumprod()
        color = cmap(idx / max(1, len(fees) - 1))
        ax.plot(wealth.index, wealth.values, linewidth=1.9, color=color, label=f"{fee} bps")

    start, end = data["hold_date"].min(), data["hold_date"].max()
    mkt = market_daily[(market_daily.index >= start) & (market_daily.index <= end)]
    ax.plot(
        mkt.index,
        (1 + mkt).cumprod(),
        color="#111827",
        linewidth=2.5,
        linestyle="--",
        label="Market proxy",
    )

    ax.axhline(1.0, color="#64748b", linewidth=0.9)
    ax.set_title(f"{model}: Window 8 daily cumulative {portfolio} returns by fee vs market")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend(title="Transaction fee", ncol=3)
    fig.tight_layout()

    out = FIGURES / f"daily_cumulative_returns_{window}_{model}_{portfolio}_fees_vs_market.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_all_models(portfolio: str = "long_short", window: str = "window8") -> list[Path]:
    daily_returns = load_or_build_daily_returns(window=window)
    start = daily_returns["hold_date"].min()
    end = daily_returns["hold_date"].max()
    market_daily = load_market_proxy(start, end)

    outputs = []
    for model in sorted(daily_returns["model"].unique()):
        outputs.append(plot_model_daily_curves(daily_returns, market_daily, model, portfolio, window))
    return outputs


def main() -> None:
    outputs = []
    for portfolio in ["long_only", "long_short"]:
        outputs.extend(plot_all_models(portfolio=portfolio, window="window8"))
    print("Saved daily cumulative return charts:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
