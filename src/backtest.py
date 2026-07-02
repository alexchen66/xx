"""
Vectorized backtest: compute portfolio returns given weights and realized returns.
"""
import pandas as pd
import numpy as np
from config import DATA_PROCESSED, COST_BPS_GRID
from portfolio import build_long_only, build_long_short, build_decile_portfolios, compute_turnover


def get_realized_returns(crsp: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """
    One-period-ahead returns on rebalance dates.
    These are the returns realized during the NEXT period after signals are generated.
    """
    # We need next-month return from each rebalance date
    labels = pd.read_parquet(DATA_PROCESSED / "labels.parquet")
    return labels[["date", "permno", "r_fwd"]].copy()


def compute_portfolio_returns(
    portfolio: pd.DataFrame,
    realized_returns: pd.DataFrame,
    cost_bps: float = 10,
) -> pd.DataFrame:
    """
    portfolio: date, permno, weight
    realized_returns: date, permno, r_fwd
    Returns: date, gross_return, net_return, turnover
    """
    turnover = compute_turnover(portfolio)

    merged = portfolio.merge(realized_returns, on=["date", "permno"], how="left")
    merged["r_fwd"] = merged["r_fwd"].fillna(0)
    merged["contribution"] = merged["weight"] * merged["r_fwd"]

    gross = merged.groupby("date")["contribution"].sum().rename("gross_return")

    results = pd.DataFrame({"gross_return": gross})
    results["turnover"] = turnover

    cost_rate = cost_bps / 10_000
    results["net_return"] = results["gross_return"] - results["turnover"] * cost_rate

    return results.reset_index()


def performance_summary(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Performance metrics from a monthly rebalance-period return series."""
    r = returns.dropna()
    monthly_ret = (1 + r).prod() ** (1 / len(r)) - 1
    ann_vol  = r.std() * np.sqrt(periods_per_year)
    sharpe   = (monthly_ret * periods_per_year) / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + r).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    max_dd   = drawdown.min()
    calmar   = (monthly_ret * periods_per_year) / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "monthly_return": round(monthly_ret, 4),
        "ann_vol":     round(ann_vol, 4),
        "sharpe":      round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "calmar":      round(calmar, 3),
        "hit_ratio":   round((r > 0).mean(), 3),
        "n_periods":   len(r),
    }


def run_backtest(predictions: pd.DataFrame, crsp: pd.DataFrame) -> dict:
    """
    Full backtest across all score columns and cost levels.
    Returns nested dict: results[model][cost_bps] = {long_only: ..., long_short: ...}
    """
    realized = pd.read_parquet(DATA_PROCESSED / "labels.parquet")[["date", "permno", "r_fwd"]]

    score_cols = [c for c in predictions.columns if c.startswith("score_")]
    results = {}

    for score_col in score_cols:
        model_name = score_col.replace("score_", "")
        print(f"\n=== Model: {model_name} ===")
        results[model_name] = {}

        port_lo = build_long_only(predictions, score_col)
        port_ls = build_long_short(predictions, score_col)

        for cost_bps in COST_BPS_GRID:
            ret_lo = compute_portfolio_returns(port_lo, realized, cost_bps)
            ret_ls = compute_portfolio_returns(port_ls, realized, cost_bps)

            results[model_name][cost_bps] = {
                "long_only":  performance_summary(ret_lo["net_return"]),
                "long_short": performance_summary(ret_ls["net_return"]),
                "avg_turnover_lo": ret_lo["turnover"].mean().round(3),
                "avg_turnover_ls": ret_ls["turnover"].mean().round(3),
            }
            if cost_bps == 10:
                lo  = results[model_name][cost_bps]["long_only"]
                ls  = results[model_name][cost_bps]["long_short"]
                print(f"  [10bps] Long-only  — Sharpe={lo['sharpe']:.2f}, "
                      f"Monthly Ret={lo['monthly_return']:.1%}, MaxDD={lo['max_drawdown']:.1%}")
                print(f"  [10bps] Long-short — Sharpe={ls['sharpe']:.2f}, "
                      f"Monthly Ret={ls['monthly_return']:.1%}, MaxDD={ls['max_drawdown']:.1%}")

    return results


def build_portfolio_return_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one row per model/cost/portfolio/rebalance date."""
    realized = pd.read_parquet(DATA_PROCESSED / "labels.parquet")[["date", "permno", "r_fwd"]]
    score_cols = [c for c in predictions.columns if c.startswith("score_")]
    rows = []

    for score_col in score_cols:
        model_name = score_col.replace("score_", "")
        portfolios = {
            "long_only": build_long_only(predictions, score_col),
            "long_short": build_long_short(predictions, score_col),
        }

        for portfolio_name, portfolio in portfolios.items():
            for cost_bps in COST_BPS_GRID:
                returns = compute_portfolio_returns(portfolio, realized, cost_bps)
                returns["model"] = model_name
                returns["cost_bps"] = cost_bps
                returns["portfolio"] = portfolio_name
                rows.append(
                    returns[
                        ["date", "model", "cost_bps", "portfolio",
                         "gross_return", "net_return", "turnover"]
                    ]
                )

    return pd.concat(rows, ignore_index=True)


def main():
    print("Loading predictions and CRSP...")
    predictions = pd.read_parquet(DATA_PROCESSED / "predictions.parquet")
    crsp        = pd.read_parquet(DATA_PROCESSED / "crsp_clean.parquet")

    results = run_backtest(predictions, crsp)

    # Save summary
    rows = []
    for model, cost_dict in results.items():
        for cost_bps, perf in cost_dict.items():
            for portfolio_type, metrics in perf.items():
                if isinstance(metrics, dict):
                    rows.append({"model": model, "cost_bps": cost_bps,
                                 "portfolio": portfolio_type, **metrics})
    summary = pd.DataFrame(rows)
    out = DATA_PROCESSED.parent / "reports" / "backtest_summary.csv"
    out.parent.mkdir(exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"\nSaved {out}")

    monthly_returns = build_portfolio_return_panel(predictions)
    monthly_out = DATA_PROCESSED.parent / "reports" / "portfolio_monthly_returns.csv"
    monthly_returns.to_csv(monthly_out, index=False)
    print(f"Saved {monthly_out}")


if __name__ == "__main__":
    main()
