"""
Risk and liquidity features: realized volatility, beta, Amihud, etc.
"""
import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_PROCESSED, DATA_FEATURES


def compute_risk_features(
    crsp: pd.DataFrame,
    ff: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    crsp = crsp[["date", "permno", "ret", "prc", "vol", "shrout", "mktcap", "dollar_vol"]].copy()
    crsp = crsp.sort_values(["permno", "date"])
    crsp["ret"] = crsp["ret"].fillna(0)

    # Merge market return for beta calculation
    crsp = crsp.merge(ff[["date", "mktrf", "rf"]], on="date", how="left")
    crsp["exc_ret"] = crsp["ret"] - crsp["rf"].fillna(0)

    grp = crsp.groupby("permno")

    # Realized volatility
    crsp["rv_20d"]  = grp["ret"].transform(lambda x: x.rolling(20,  min_periods=15).std())
    crsp["rv_60d"]  = grp["ret"].transform(lambda x: x.rolling(60,  min_periods=45).std())
    crsp["rv_120d"] = grp["ret"].transform(lambda x: x.rolling(120, min_periods=90).std())

    # Downside volatility (semi-deviation)
    crsp["downside_vol_60d"] = grp["ret"].transform(
        lambda x: x.rolling(60, min_periods=45).apply(
            lambda r: r[r < 0].std() if (r < 0).sum() > 5 else np.nan, raw=True
        )
    )

    # Rolling beta (252d OLS: exc_ret ~ mktrf)
    def rolling_beta(df, window=252, min_obs=120):
        exc = df["exc_ret"].values
        mkt = df["mktrf"].values
        betas = np.full(len(exc), np.nan)
        for i in range(window, len(exc) + 1):
            y = exc[i - window:i]
            x = mkt[i - window:i]
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < min_obs:
                continue
            x_m, y_m = x[mask], y[mask]
            cov = np.cov(x_m, y_m)
            betas[i - 1] = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else np.nan
        return betas

    beta_list = []
    for permno, g in crsp.groupby("permno"):
        b = rolling_beta(g)
        beta_list.append(pd.Series(b, index=g.index))
    crsp["beta_252d"] = pd.concat(beta_list).reindex(crsp.index)

    # Idiosyncratic volatility: std of residuals from mktrf regression (252d)
    def rolling_idio_vol(df, window=252, min_obs=120):
        exc = df["exc_ret"].values
        mkt = df["mktrf"].values
        ivol = np.full(len(exc), np.nan)
        for i in range(window, len(exc) + 1):
            y = exc[i - window:i]
            x = mkt[i - window:i]
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < min_obs:
                continue
            x_m, y_m = x[mask], y[mask]
            beta = np.cov(x_m, y_m)[0, 1] / np.var(x_m) if np.var(x_m) > 0 else 0
            resid = y_m - beta * x_m
            ivol[i - 1] = resid.std()
        return ivol

    ivol_list = []
    for permno, g in crsp.groupby("permno"):
        iv = rolling_idio_vol(g)
        ivol_list.append(pd.Series(iv, index=g.index))
    crsp["idio_vol_252d"] = pd.concat(ivol_list).reindex(crsp.index)

    # Max drawdown
    def rolling_max_dd(price_series, window):
        def max_dd(w):
            if len(w) < 2:
                return np.nan
            peak = np.maximum.accumulate(w)
            dd = (w - peak) / peak
            return dd.min()
        return price_series.rolling(window, min_periods=int(window * 0.8)).apply(max_dd, raw=True)

    crsp["max_dd_60d"]  = grp["prc"].transform(lambda x: rolling_max_dd(x, 60))
    crsp["max_dd_252d"] = grp["prc"].transform(lambda x: rolling_max_dd(x, 252))

    # Rolling skewness
    crsp["skew_60d"] = grp["ret"].transform(
        lambda x: x.rolling(60, min_periods=45).skew()
    )

    # Liquidity features
    crsp["turnover_20d"] = grp["vol"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    ) / (crsp["shrout"] * 1000).replace(0, np.nan)

    crsp["dollar_vol_20d"] = grp["dollar_vol"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )

    # Amihud illiquidity: mean(|ret| / dollar_vol)
    crsp["amihud_daily"] = crsp["ret"].abs() / crsp["dollar_vol"].replace(0, np.nan)
    crsp["amihud_20d"] = grp["amihud_daily"].transform(
        lambda x: x.rolling(20, min_periods=15).mean()
    )

    feature_cols = [
        "rv_20d", "rv_60d", "rv_120d", "downside_vol_60d",
        "beta_252d", "idio_vol_252d",
        "max_dd_60d", "max_dd_252d", "skew_60d",
        "amihud_20d", "dollar_vol_20d", "turnover_20d",
    ]

    rebal_pairs = universe[["date", "permno"]].copy()
    result = rebal_pairs.merge(
        crsp[["date", "permno"] + feature_cols],
        on=["date", "permno"],
        how="left",
    )
    return result


def main():
    print("Loading data...")
    crsp     = pd.read_parquet(DATA_PROCESSED / "crsp_clean.parquet")
    ff       = pd.read_parquet(DATA_RAW / "ff_factors_daily.parquet")
    universe = pd.read_parquet(DATA_PROCESSED / "universe.parquet")

    print("Computing risk/liquidity features (beta and idio_vol are slow)...")
    features = compute_risk_features(crsp, ff, universe)

    out = DATA_FEATURES / "features_risk.parquet"
    features.to_parquet(out, index=False)
    print(f"  Saved {out} — {len(features):,} rows")


if __name__ == "__main__":
    main()
