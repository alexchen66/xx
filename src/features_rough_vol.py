"""
Rough volatility features combining two complementary approaches:

1. Whittle estimator for H (statistically efficient, spectral MLE):
   fits f(lambda) = C_H * |lambda|^(1-2H) to the periodogram of log-RV.
   Asymptotically efficient; achieves Cramér-Rao bound for H estimation.

2. Volterra kernel filtered variance (more robust, no H estimation needed):
   Instead of estimating H, pre-compute kernel weights at a fixed H grid
   {0.05, 0.10, 0.20, 0.50} and apply them as a causal weighted average of
   squared returns. The "roughness spread" log(X^{H=0.05}) - log(X^{H=0.50})
   captures how much more a rough kernel weights recent variance shocks relative
   to a smooth kernel — a direct data-driven roughness signal.

   w_ℓ(H) = [ℓ^α - (ℓ-1)^α] / Γ(α+1),   α = H + 0.5
   X^{H,L}_{t} = Σ_{ℓ=1}^{L} w_ℓ(H) · q_{t-ℓ}     (q = squared return)

Reference: Giraitis, Koul, Surgailis (2012); Gatheral, Jaisson, Rosenbaum (2018);
           El Euch & Rosenbaum (2019).
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import periodogram
from scipy.special import gamma as _gamma
from config import DATA_RAW, DATA_PROCESSED, DATA_FEATURES


# ---------------------------------------------------------------------------
# Whittle estimator for the Hurst exponent
# ---------------------------------------------------------------------------

def whittle_hurst(x: np.ndarray, freq_trim: float = 0.05) -> float:
    """
    Estimate Hurst exponent H of a fractional Brownian motion (or fGn) series
    using the Whittle (spectral likelihood) estimator.

    For a process with spectral density f(lambda) = C_H * lambda^(1-2H):
        L(H) = sum_j [ log f(lambda_j; H) + I(lambda_j) / f(lambda_j; H) ]
             = sum_j [ (1-2H) log lambda_j + I(lambda_j) / (C_H * lambda_j^(1-2H)) ]

    Minimizing over H gives the Whittle MLE. C_H is profiled out analytically.

    Parameters
    ----------
    x : array-like, the log-RV time series (stationary increments assumed)
    freq_trim : fraction of lowest frequencies to exclude (avoids trend contamination)

    Returns
    -------
    H : float in (0, 1), or np.nan if estimation fails
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return np.nan

    # Use first differences to make the process stationary (fGn from fBm)
    dx = np.diff(x)
    n = len(dx)

    # Periodogram of dx
    freqs, Ix = periodogram(dx, window="boxcar", scaling="density")

    # Exclude zero frequency and trim low frequencies (avoid non-stationarity artifacts)
    trim = max(1, int(freq_trim * n))
    freqs = freqs[trim:]
    Ix    = Ix[trim:]

    # Only use positive frequencies
    mask = freqs > 0
    freqs = freqs[mask]
    Ix    = Ix[mask]

    if len(freqs) < 5:
        return np.nan

    log_freqs = np.log(freqs)

    def neg_whittle_loglik(H):
        # Spectral density of fGn is proportional to |lambda|^(1-2H)
        # up to a constant C(H) which we profile out
        alpha = 1 - 2 * H   # spectral exponent
        log_f = alpha * log_freqs   # log f(lambda) up to log(C_H)

        # Profile out C_H: C_H_hat = mean(I / lambda^alpha)
        f_unnorm = np.exp(log_f)
        C_H = np.mean(Ix / f_unnorm)
        if C_H <= 0:
            return 1e10

        # Whittle log-likelihood (negated for minimization)
        ll = np.mean(log_f + np.log(C_H) + Ix / (C_H * f_unnorm))
        return ll

    result = minimize_scalar(
        neg_whittle_loglik,
        bounds=(0.01, 0.99),
        method="bounded",
        options={"xatol": 1e-4},
    )

    if not result.success and result.fun > 1e9:
        return np.nan

    H = result.x
    return float(H) if 0 < H < 1 else np.nan


def ols_scaling_hurst(log_rv: np.ndarray, lags=(2, 4, 8, 16, 32), q: int = 2) -> float:
    """
    Fallback OLS scaling-moment estimator for H.
    E[|X(t+lag) - X(t)|^q] ~ lag^(qH)
    Used for comparison / robustness check against the Whittle estimator.
    """
    x = np.asarray(log_rv, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max(lags) + 10:
        return np.nan

    log_lags, log_moments = [], []
    for lag in lags:
        diffs = np.abs(x[lag:] - x[:-lag]) ** q
        m = np.mean(diffs[np.isfinite(diffs)])
        if m > 0:
            log_lags.append(np.log(lag))
            log_moments.append(np.log(m))

    if len(log_lags) < 3:
        return np.nan

    X = np.array(log_lags)
    Y = np.array(log_moments)
    slope = np.cov(X, Y)[0, 1] / np.var(X)
    H = slope / q
    return float(H) if 0 < H < 1 else np.nan


# ---------------------------------------------------------------------------
# Volterra kernel filtered variance
# ---------------------------------------------------------------------------

# Pre-set H grid: {0.05, 0.10, 0.20, 0.50}
# 0.50 is the smooth (Heston-like) baseline; lower H = rougher
VOLTERRA_H_GRID = [0.05, 0.10, 0.20, 0.50]
VOLTERRA_WINDOW = 126  # days


def _volterra_weights(H: float, L: int) -> np.ndarray:
    """
    Bin-integrated Volterra kernel weights of length L.
    w_ℓ(H) = [ℓ^α - (ℓ-1)^α] / Γ(α+1),  α = H + 0.5,  ℓ = 1, ..., L

    For H < 0.5 (rough): w_1 >> w_L  — heavy weight on recent shocks.
    For H = 0.5 (BM):    uniform-ish weights.
    For H > 0.5 (smooth): slowly increasing with lag.

    Returns array ordered [w_1, w_2, ..., w_L], normalized to sum = 1.
    """
    alpha = H + 0.5
    ell = np.arange(1, L + 1, dtype=float)
    w = (ell ** alpha - (ell - 1) ** alpha) / _gamma(alpha + 1)
    return w / w.sum()


def _volterra_filtered_var(q: np.ndarray, H: float, L: int,
                            min_coverage: float = 0.70) -> np.ndarray:
    """
    Vectorized causal Volterra-kernel weighted variance via np.convolve.

    X[t] = Σ_{ℓ=1}^{L} w_ℓ(H) · q[t-ℓ]   for t >= L+1

    Uses np.convolve for C-speed; handles NaN via separate coverage array.
    """
    w = _volterra_weights(H, L)
    N = len(q)

    q_nan    = np.isfinite(q).astype(float)
    q_filled = np.where(np.isfinite(q), q, 0.0)

    # convolve then slice: conv[t-1] = Σ_j weights[j]*q[t-1-j]
    #   = w_1*q[t-1] + w_2*q[t-2] + ... + w_L*q[t-L]  ✓  (lags 1..L)
    conv_val = np.convolve(q_filled, w, mode="full")[L - 1: N - 1]
    conv_wt  = np.convolve(q_nan,    w, mode="full")[L - 1: N - 1]

    result = np.full(N, np.nan)
    ok = conv_wt >= min_coverage  # weights sum to 1; require 70% coverage
    if ok.any():
        result[L:][ok] = conv_val[ok] / conv_wt[ok]  # normalize for missing days
    return result


def compute_volterra_features(
    crsp: pd.DataFrame,
    universe: pd.DataFrame,
    h_grid: list = VOLTERRA_H_GRID,
    window: int = VOLTERRA_WINDOW,
    use_idio: bool = True,
) -> pd.DataFrame:
    """
    Compute Volterra-kernel filtered variance features for all stocks.

    For each H in h_grid:
      volterra_var_H{H*100:03.0f}_{window}d   : total-return version
      idio_volterra_var_H{H*100:03.0f}_{window}d : idiosyncratic residual version

    Roughness spread features (log ratio vs smooth baseline H=0.50):
      volterra_rough_spread_H005_H050_{window}d
      volterra_rough_spread_H010_H050_{window}d
      (and idio equivalents)
    """
    df = crsp[["date", "permno", "ret"]].copy()
    df = df.sort_values(["permno", "date"])
    df["q"] = df["ret"].fillna(0) ** 2  # squared returns

    # Idiosyncratic squared returns (residual after stripping market via 60d beta)
    if use_idio and "idio_ret" in crsp.columns:
        df["q_idio"] = crsp["idio_ret"].fillna(0) ** 2
    elif use_idio:
        # Fallback: use total return squared (no idio decomposition available)
        df["q_idio"] = df["q"]

    results = {}
    for permno, g in df.groupby("permno", sort=False):
        entry = {"date": g["date"].values, "permno": permno}
        q      = g["q"].values
        q_idio = g["q_idio"].values if use_idio else q

        for H_val in h_grid:
            tag = f"H{int(round(H_val * 100)):03d}"
            entry[f"volterra_var_{tag}_{window}d"] = _volterra_filtered_var(q, H_val, window)
            if use_idio:
                entry[f"idio_volterra_var_{tag}_{window}d"] = _volterra_filtered_var(q_idio, H_val, window)

        results[permno] = pd.DataFrame(entry)

    panel = pd.concat(results.values(), ignore_index=True)

    # Roughness spreads: log(X^rough) - log(X^smooth)
    baseline = f"volterra_var_H050_{window}d"
    log_base  = np.log(panel[baseline].clip(lower=1e-10))
    for H_val in [0.05, 0.10]:
        tag = f"H{int(round(H_val * 100)):03d}"
        rough_col = f"volterra_var_{tag}_{window}d"
        panel[f"volterra_rough_spread_{tag}_H050_{window}d"] = (
            np.log(panel[rough_col].clip(lower=1e-10)) - log_base
        )

    if use_idio:
        idio_base = f"idio_volterra_var_H050_{window}d"
        log_idio_base = np.log(panel[idio_base].clip(lower=1e-10))
        for H_val in [0.05, 0.10]:
            tag = f"H{int(round(H_val * 100)):03d}"
            rough_col = f"idio_volterra_var_{tag}_{window}d"
            panel[f"idio_volterra_rough_spread_{tag}_H050_{window}d"] = (
                np.log(panel[rough_col].clip(lower=1e-10)) - log_idio_base
            )

    rebal_pairs = universe[["date", "permno"]].copy()
    volterra_cols = [c for c in panel.columns if c not in ("date", "permno")]
    return rebal_pairs.merge(panel[["date", "permno"] + volterra_cols],
                             on=["date", "permno"], how="left")


# ---------------------------------------------------------------------------
# Rolling Hurst estimation on log-RV series
# ---------------------------------------------------------------------------

def compute_all_rough_features(
    crsp: pd.DataFrame,
    ff: pd.DataFrame,
    universe: pd.DataFrame,
    rv_window: int = 20,
    hurst_windows: tuple = (126, 252),
    volterra_h_grid: list = None,
    volterra_window: int = VOLTERRA_WINDOW,
) -> pd.DataFrame:
    """
    Compute all rough volatility features with two-stage design:

    Stage 1 — vectorized daily features (pandas C-speed):
        rv, log_rv, rolling beta, idio_ret, idio_log_rv, vol-of-vol, Volterra

    Stage 2 — Whittle/OLS called ONLY at rebalance dates (not every trading day):
        745K calls instead of tens of millions → ~15 min total vs. 10+ hours
    """
    if volterra_h_grid is None:
        volterra_h_grid = VOLTERRA_H_GRID

    # --- Prep ---
    universe_permnos = set(universe["permno"].unique())
    crsp = crsp[["date", "permno", "ret"]].copy()
    crsp = crsp[crsp["permno"].isin(universe_permnos)]
    crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)
    crsp["ret"] = crsp["ret"].fillna(0)

    # Merge FF for beta computation
    crsp = crsp.merge(ff[["date", "mktrf"]], on="date", how="left")
    crsp["mktrf"] = crsp["mktrf"].fillna(0)

    print("  Stage 1: vectorized daily features...", flush=True)

    # Realized vol & log-RV (C-speed rolling std)
    crsp["rv_20d"] = crsp.groupby("permno")["ret"].transform(
        lambda x: x.rolling(rv_window, min_periods=15).std()
    )
    crsp["log_rv"] = np.where(crsp["rv_20d"] > 0, np.log(crsp["rv_20d"]), np.nan)

    # Rolling 60d beta -> idio_ret via rolling moments. Avoid groupby.apply,
    # which is both slower and has changed grouping-column behavior in newer pandas.
    crsp["xy_60"] = crsp["ret"] * crsp["mktrf"]
    crsp["x2_60"] = crsp["mktrf"] ** 2
    grp = crsp.groupby("permno")
    mean_x = grp["mktrf"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    mean_y = grp["ret"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    mean_xy = grp["xy_60"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    mean_x2 = grp["x2_60"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    var_x = (mean_x2 - mean_x ** 2).clip(lower=1e-10)
    beta = (mean_xy - mean_x * mean_y) / var_x
    crsp["idio_ret"] = crsp["ret"] - beta * crsp["mktrf"]

    crsp["idio_rv_20d"] = crsp.groupby("permno")["idio_ret"].transform(
        lambda x: x.rolling(rv_window, min_periods=15).std()
    )
    crsp["idio_log_rv"] = np.where(crsp["idio_rv_20d"] > 0,
                                    np.log(crsp["idio_rv_20d"]), np.nan)

    # Vol-of-vol (vectorized)
    crsp["dlog_rv"] = crsp.groupby("permno")["log_rv"].transform(lambda x: x.diff())
    crsp["vol_of_vol_60d"]  = crsp.groupby("permno")["dlog_rv"].transform(
        lambda x: x.rolling(60, min_periods=45).std()
    )
    crsp["vol_of_vol_126d"] = crsp.groupby("permno")["dlog_rv"].transform(
        lambda x: x.rolling(126, min_periods=90).std()
    )

    print("  Stage 1 done. Stage 2: Whittle + Volterra at rebalance dates...", flush=True)

    # Rebalance dates per stock
    rebal_by_permno = (
        universe.groupby("permno")["date"]
        .apply(lambda x: np.sort(x.values))
        .to_dict()
    )

    all_records = []
    permnos = crsp["permno"].unique()
    n = len(permnos)

    for i, permno in enumerate(permnos):
        if i % 2000 == 0:
            print(f"  {i}/{n} stocks...", flush=True)

        rdates = rebal_by_permno.get(permno)
        if rdates is None or len(rdates) == 0:
            continue

        g = crsp[crsp["permno"] == permno]
        if len(g) < 30:
            continue

        dates        = g["date"].values
        log_rv_arr   = g["log_rv"].values
        idio_lrv_arr = g["idio_log_rv"].values
        ret_arr      = g["ret"].values
        idio_arr     = g["idio_ret"].fillna(0).values
        vov60_arr    = g["vol_of_vol_60d"].values
        vov126_arr   = g["vol_of_vol_126d"].values

        # Volterra (convolve per stock — fast)
        q      = ret_arr ** 2
        q_idio = idio_arr ** 2
        v_arrays = {}
        log_base = log_idio_base = None
        for H_val in volterra_h_grid:
            tag = f"H{int(round(H_val * 100)):03d}"
            xv = _volterra_filtered_var(q,      H_val, volterra_window)
            xi = _volterra_filtered_var(q_idio, H_val, volterra_window)
            v_arrays[f"volterra_var_{tag}_{volterra_window}d"]      = xv
            v_arrays[f"idio_volterra_var_{tag}_{volterra_window}d"] = xi
            if H_val == 0.50:
                log_base      = np.log(np.clip(xv, 1e-10, None))
                log_idio_base = np.log(np.clip(xi, 1e-10, None))
        for H_val in [0.05, 0.10]:
            tag = f"H{int(round(H_val * 100)):03d}"
            xv = v_arrays[f"volterra_var_{tag}_{volterra_window}d"]
            xi = v_arrays[f"idio_volterra_var_{tag}_{volterra_window}d"]
            v_arrays[f"volterra_rough_spread_{tag}_H050_{volterra_window}d"] = (
                np.log(np.clip(xv, 1e-10, None)) - log_base
            )
            v_arrays[f"idio_volterra_rough_spread_{tag}_H050_{volterra_window}d"] = (
                np.log(np.clip(xi, 1e-10, None)) - log_idio_base
            )

        # Whittle + OLS only at rebalance dates
        rows = []
        for rdate in rdates:
            idx = np.searchsorted(dates, rdate)
            if idx >= len(dates) or dates[idx] != rdate:
                continue

            row = {
                "date":           rdate,
                "permno":         permno,
                "vol_of_vol_60d": vov60_arr[idx],
                "vol_of_vol_126d":vov126_arr[idx],
            }

            # Volterra at this index
            for col, arr in v_arrays.items():
                row[col] = arr[idx] if idx < len(arr) else np.nan

            # Whittle/OLS: extract trailing window and call once
            for win in hurst_windows:
                mo   = int(win * 0.7)
                s    = max(0, idx - win + 1)
                w    = log_rv_arr[s: idx + 1]
                w    = w[np.isfinite(w)]
                h    = whittle_hurst(w)     if len(w) >= mo else np.nan
                h_ol = ols_scaling_hurst(w) if len(w) >= mo else np.nan
                row[f"hurst_{win}d"]     = h
                row[f"roughness_{win}d"] = (0.5 - h) if not np.isnan(h) else np.nan
                row[f"hurst_ols_{win}d"] = h_ol

            # Idio Whittle (126d only)
            mo126 = int(126 * 0.7)
            s126  = max(0, idx - 126 + 1)
            wi    = idio_lrv_arr[s126: idx + 1]
            wi    = wi[np.isfinite(wi)]
            h_i   = whittle_hurst(wi) if len(wi) >= mo126 else np.nan
            row["idio_hurst_126d"]    = h_i
            row["idio_roughness_126d"] = (0.5 - h_i) if not np.isnan(h_i) else np.nan

            rows.append(row)

        if rows:
            all_records.append(pd.DataFrame(rows))

    result = pd.concat(all_records, ignore_index=True)
    result["roughness_x_momentum"]    = np.nan
    result["roughness_x_illiquidity"] = np.nan
    return result


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    print("Loading data...", flush=True)
    # Load only columns needed — crsp_clean has many columns; keep small footprint
    crsp = pd.read_parquet(
        DATA_PROCESSED / "crsp_clean.parquet",
        columns=["date", "permno", "ret", "mktcap", "siccd"],
    )
    ff       = pd.read_parquet(DATA_RAW / "ff_factors_daily.parquet")
    universe = pd.read_parquet(DATA_PROCESSED / "universe.parquet")
    print(f"  crsp {len(crsp):,} rows | ff {len(ff):,} | universe {len(universe):,}", flush=True)

    print("Computing Whittle + Volterra features in one pass (~30 min)...", flush=True)
    features = compute_all_rough_features(crsp, ff, universe)

    # Merge price + risk features for interaction terms
    price_f = pd.read_parquet(DATA_FEATURES / "features_price.parquet")
    risk_f  = pd.read_parquet(DATA_FEATURES / "features_risk.parquet")
    features = features.merge(price_f[["date", "permno", "mom_12_1"]],  on=["date", "permno"], how="left")
    features = features.merge(risk_f[["date",  "permno", "amihud_20d"]], on=["date", "permno"], how="left")
    features["roughness_x_momentum"]    = features["roughness_126d"] * features["mom_12_1"]
    features["roughness_x_illiquidity"] = features["roughness_126d"] * features["amihud_20d"]
    features = features.drop(columns=["mom_12_1", "amihud_20d"])

    out = DATA_FEATURES / "features_rough_vol.parquet"
    features.to_parquet(out, index=False)
    print(f"  Saved {out} — {len(features):,} rows, {len(features.columns)} columns")

    # Sanity checks
    h = features["roughness_126d"].dropna()
    print(f"  roughness_126d  : mean={h.mean():.3f}  std={h.std():.3f}  "
          f"range=[{h.min():.3f}, {h.max():.3f}]  coverage={h.notna().mean():.1%}")
    s = features["volterra_rough_spread_H005_H050_126d"].dropna()
    print(f"  volterra_spread : mean={s.mean():.3f}  std={s.std():.3f}  "
          f"range=[{s.min():.3f}, {s.max():.3f}]  coverage={s.notna().mean():.1%}")


if __name__ == "__main__":
    main()
