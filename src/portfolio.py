"""
Portfolio construction from model scores.
"""
import pandas as pd
import numpy as np
from config import TOP_PCT


def _equal_weights(n: int) -> np.ndarray:
    """Equal weights summing to 1."""
    return np.repeat(1.0 / n, n)


def _rank_weights(n: int) -> np.ndarray:
    """Linear rank weights, largest for the first row and summing to 1."""
    raw = np.arange(n, 0, -1, dtype=float)
    return raw / raw.sum()


def _weights(n: int, scheme: str = "equal") -> np.ndarray:
    if scheme == "equal":
        return _equal_weights(n)
    if scheme == "rank":
        return _rank_weights(n)
    raise ValueError(f"Unsupported weighting scheme: {scheme}")


def build_long_only(
    predictions: pd.DataFrame,
    score_col: str,
    top_pct: float = TOP_PCT,
    weighting: str = "equal",
) -> pd.DataFrame:
    """Build top-decile long-only portfolios by score on each rebalance date."""
    portfolios = []
    for date, df in predictions.groupby("date"):
        n = max(1, int(len(df) * top_pct))
        df = df.nlargest(n, score_col).copy()
        df["weight"] = _weights(len(df), weighting)
        df["side"]   = "long"
        portfolios.append(df[["date", "permno", "weight", "side"]])
    return pd.concat(portfolios, ignore_index=True)


def build_long_short(
    predictions: pd.DataFrame,
    score_col: str,
    top_pct: float = TOP_PCT,
    weighting: str = "equal",
) -> pd.DataFrame:
    """Build top-decile long and bottom-decile short portfolios."""
    portfolios = []
    for date, df in predictions.groupby("date"):
        df = df.sort_values(score_col, ascending=False).copy()
        n = max(1, int(len(df) * top_pct))

        long_df  = df.head(n).copy()
        short_df = df.tail(n).copy()

        long_weights = _weights(n, weighting)
        short_weights = _weights(n, weighting)
        if weighting == "rank":
            short_weights = short_weights[::-1]

        long_df["weight"] = long_weights
        short_df["weight"] = -short_weights
        long_df["side"]    = "long"
        short_df["side"]   = "short"

        portfolios.append(
            pd.concat([
                long_df[["date", "permno", "weight", "side"]],
                short_df[["date", "permno", "weight", "side"]],
            ])
        )
    return pd.concat(portfolios, ignore_index=True)


def build_decile_portfolios(
    predictions: pd.DataFrame,
    score_col: str,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Build all decile portfolios for decile-return analysis."""
    records = []
    for date, df in predictions.groupby("date"):
        df = df.copy()
        df["decile"] = pd.qcut(df[score_col], q=n_deciles, labels=False, duplicates="drop")
        records.append(df[["date", "permno", "decile"]])
    return pd.concat(records, ignore_index=True)


def compute_turnover(portfolio: pd.DataFrame) -> pd.Series:
    """
    Turnover per period: sum(|w_t - w_{t-1}|).
    Stocks entering/leaving the portfolio count as full weight change.
    """
    dates = sorted(portfolio["date"].unique())
    turnovers = []

    prev_weights = {}
    for date in dates:
        curr = portfolio[portfolio["date"] == date].set_index("permno")["weight"].to_dict()

        all_permnos = set(prev_weights) | set(curr)
        to = sum(abs(curr.get(p, 0) - prev_weights.get(p, 0)) for p in all_permnos)
        turnovers.append({"date": date, "turnover": to})

        prev_weights = curr

    return pd.DataFrame(turnovers).set_index("date")["turnover"]
