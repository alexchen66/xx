"""
Evaluation metrics: Rank IC, ICIR, decile spreads, and roughness-specific diagnostics.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from config import DATA_PROCESSED
from portfolio import build_decile_portfolios


def rank_ic_series(predictions: pd.DataFrame, score_col: str, label_col: str = "y_xs") -> pd.Series:
    """Spearman correlation between score and future return, per rebalance date."""
    def _ic(df):
        r, _ = spearmanr(df[score_col], df[label_col], nan_policy="omit")
        return r
    return predictions.groupby("date").apply(_ic).rename("rank_ic")


def ic_summary(ic: pd.Series) -> dict:
    return {
        "ic_mean":     round(ic.mean(), 4),
        "ic_std":      round(ic.std(),  4),
        "icir":        round(ic.mean() / ic.std(), 3) if ic.std() > 0 else np.nan,
        "hit_ratio":   round((ic > 0).mean(), 3),
        "ic_positive_months": int((ic > 0).sum()),
        "n_months":    len(ic),
    }


def decile_return_analysis(
    predictions: pd.DataFrame,
    score_col: str,
    label_col: str = "r_fwd",
) -> pd.DataFrame:
    """
    For each rebalance date, assign deciles and compute mean return per decile.
    Returns: decile × metric table.
    """
    deciles = build_decile_portfolios(predictions, score_col)
    merged  = deciles.merge(predictions[["date", "permno", label_col]], on=["date", "permno"])

    stats = (
        merged.groupby("decile")[label_col]
        .agg(mean_return="mean", std="std", count="count")
        .reset_index()
    )
    stats["mean_return_ann"] = stats["mean_return"] * 12
    return stats


def top_bottom_spread(predictions: pd.DataFrame, score_col: str, label_col: str = "r_fwd") -> pd.Series:
    """Monthly top-decile minus bottom-decile return spread."""
    def _spread(df):
        n = max(1, int(len(df) * 0.1))
        df_sorted = df.sort_values(score_col)
        return df_sorted[label_col].tail(n).mean() - df_sorted[label_col].head(n).mean()
    return predictions.groupby("date").apply(_spread).rename("top_bottom_spread")


def roughness_diagnostics(
    predictions: pd.DataFrame,
    roughness_col: str = "roughness_126d",
    label_col: str = "r_fwd",
) -> dict:
    """
    Standalone roughness factor diagnostics:
    - Single-factor Rank IC
    - Correlation with realized vol, size, liquidity
    - Decile return monotonicity
    """
    if roughness_col not in predictions.columns:
        return {}

    ic = rank_ic_series(predictions, roughness_col, label_col)

    corr_cols = ["rv_20d", "mktcap", "amihud_20d", "beta_252d"]
    corr_cols = [c for c in corr_cols if c in predictions.columns]
    correlations = {
        f"corr_{c}": round(predictions[[roughness_col, c]].corr().iloc[0, 1], 3)
        for c in corr_cols
    }

    deciles = decile_return_analysis(predictions, roughness_col, label_col)
    monotone = bool(
        deciles.sort_values("decile")["mean_return"].is_monotonic_increasing
        or deciles.sort_values("decile")["mean_return"].is_monotonic_decreasing
    )

    return {
        "single_factor_ic": ic_summary(ic),
        "correlations_with_controls": correlations,
        "decile_monotonicity": monotone,
        "decile_returns": deciles.to_dict("records"),
    }


def full_evaluation(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Run IC analysis for all score columns. Return a summary DataFrame.
    """
    score_cols = [c for c in predictions.columns if c.startswith("score_")]
    rows = []
    for score_col in score_cols:
        model_name = score_col.replace("score_", "")
        ic = rank_ic_series(predictions, score_col)
        summary = ic_summary(ic)
        spread  = top_bottom_spread(predictions, score_col)
        summary["model"] = model_name
        summary["spread_mean"] = round(spread.mean(), 4)
        summary["spread_icir"] = round(spread.mean() / spread.std(), 3) if spread.std() > 0 else np.nan
        rows.append(summary)
        print(f"\n{model_name}:")
        print(f"  IC Mean={summary['ic_mean']:.4f}, ICIR={summary['icir']:.3f}, "
              f"Hit={summary['hit_ratio']:.1%}, Spread={summary['spread_mean']:.4f}")
    return pd.DataFrame(rows)


def main():
    print("Loading predictions...")
    predictions = pd.read_parquet(DATA_PROCESSED / "predictions.parquet")

    # Merge r_fwd from labels (needed for spread and decile return analysis)
    labels = pd.read_parquet(DATA_PROCESSED / "labels.parquet")
    predictions = predictions.merge(
        labels[["date", "permno", "r_fwd"]], on=["date", "permno"], how="left"
    )

    print("\n=== IC Analysis ===")
    summary = full_evaluation(predictions)

    out = DATA_PROCESSED.parent / "reports"
    out.mkdir(exist_ok=True)
    summary.to_csv(out / "ic_summary.csv", index=False)

    print("\n=== Roughness Single-Factor Diagnostics ===")
    # Need to merge roughness features back for standalone analysis
    from config import DATA_FEATURES
    features = pd.read_parquet(DATA_FEATURES / "features_rough_vol.parquet")
    labels   = pd.read_parquet(DATA_PROCESSED / "labels.parquet")
    rough_df = features.merge(labels[["date", "permno", "r_fwd"]], on=["date", "permno"])

    if "roughness_126d" in rough_df.columns:
        diag = roughness_diagnostics(rough_df)
        ic_d = diag.get("single_factor_ic", {})
        print(f"  Roughness standalone IC Mean={ic_d.get('ic_mean', 'N/A'):.4f}, "
              f"ICIR={ic_d.get('icir', 'N/A'):.3f}")

    print(f"\nSaved ic_summary.csv to reports/")


if __name__ == "__main__":
    main()
