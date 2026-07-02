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

    # Downside volatility (semi-deviation). This vectorized proxy uses negative
    # returns only and avoids a slow rolling Python callback on the full CRSP panel.
    crsp["ret_neg"] = crsp["ret"].where(crsp["ret"] < 0)
    crsp["downside_vol_60d"] = grp["ret_neg"].transform(
        lambda x: x.rolling(60, min_periods=15).std()
    )

    # Rolling beta and idiosyncratic volatility via rolling moments.
    crsp["mktrf"] = crsp["mktrf"].fillna(0)
    crsp["xy"] = crsp["exc_ret"] * crsp["mktrf"]
    crsp["x2"] = crsp["mktrf"] ** 2
    crsp["y2"] = crsp["exc_ret"] ** 2

    mean_x = grp["mktrf"].transform(lambda x: x.rolling(252, min_periods=120).mean())
    mean_y = grp["exc_ret"].transform(lambda x: x.rolling(252, min_periods=120).mean())
    mean_xy = grp["xy"].transform(lambda x: x.rolling(252, min_periods=120).mean())
    mean_x2 = grp["x2"].transform(lambda x: x.rolling(252, min_periods=120).mean())
    mean_y2 = grp["y2"].transform(lambda x: x.rolling(252, min_periods=120).mean())

    var_x = (mean_x2 - mean_x ** 2).clip(lower=1e-12)
    var_y = (mean_y2 - mean_y ** 2).clip(lower=0)
    cov_xy = mean_xy - mean_x * mean_y
    crsp["beta_252d"] = cov_xy / var_x
    resid_var = (var_y - (cov_xy ** 2) / var_x).clip(lower=0)
    crsp["idio_vol_252d"] = np.sqrt(resid_var)

    # Max drawdown proxy using rolling peak and rolling drawdown minimum.
    roll_peak_60 = grp["prc"].transform(lambda x: x.rolling(60, min_periods=48).max())
    roll_peak_252 = grp["prc"].transform(lambda x: x.rolling(252, min_periods=202).max())
    crsp["dd_60d"] = crsp["prc"] / roll_peak_60 - 1
    crsp["dd_252d"] = crsp["prc"] / roll_peak_252 - 1
    crsp["max_dd_60d"] = grp["dd_60d"].transform(lambda x: x.rolling(60, min_periods=48).min())
    crsp["max_dd_252d"] = grp["dd_252d"].transform(lambda x: x.rolling(252, min_periods=202).min())

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
