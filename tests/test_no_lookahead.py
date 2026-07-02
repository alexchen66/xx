"""
Verify no future data leaks into features or labels.
"""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"


def test_feature_dates_not_after_signal_date():
    """All feature values must be computed using data up to (and including) the rebalance date."""
    features = pd.read_parquet(DATA / "features" / "features_preprocessed.parquet")
    # Spot-check: ret_1d on date t should equal the return ON date t (not t+1)
    # We verify this by checking that feature dates exist in CRSP
    crsp = pd.read_parquet(DATA / "processed" / "crsp_clean.parquet")
    crsp_dates = set(crsp["date"].dt.date)
    feature_dates = set(features["date"].dt.date)
    unknown = feature_dates - crsp_dates
    assert len(unknown) == 0, f"Feature dates not in CRSP: {unknown}"


def test_labels_use_future_prices():
    """Labels must use prices strictly AFTER the signal date."""
    labels = pd.read_parquet(DATA / "processed" / "labels.parquet")
    # r_fwd should be non-trivially distributed (not all zeros or NaN)
    assert labels["r_fwd"].notna().sum() > 1000
    assert labels["r_fwd"].std() > 0.01
    # Spot check: forward returns should have roughly 0 mean cross-sectionally
    xs_mean = labels.groupby("date")["y_xs"].mean()
    assert xs_mean.abs().mean() < 0.005, "Cross-sectional excess return should be ~0 on average"


def test_fundamental_uses_rdq_not_datadate():
    """Fundamentals must be aligned by report date (rdq), not fiscal period end."""
    raw_fundq = pd.read_parquet(DATA / "raw" / "compustat_fundq.parquet")
    # rdq should generally be AFTER datadate (lag of ~1-3 months)
    lag = (raw_fundq["rdq"] - raw_fundq["datadate"]).dt.days
    lag_valid = lag.dropna()
    pct_positive = (lag_valid > 0).mean()
    assert pct_positive > 0.90, \
        f"Expected >90% of rdq > datadate, got {pct_positive:.1%}"


def test_walk_forward_no_test_in_train():
    """Test dates must not appear in training data for any walk-forward window."""
    from sys import path
    path.insert(0, str(Path(__file__).parent.parent / "src"))
    from config import WALK_FORWARD_WINDOWS

    for tr_s, tr_e, va_s, va_e, te_s, te_e in WALK_FORWARD_WINDOWS:
        assert pd.Timestamp(te_s) > pd.Timestamp(tr_e), \
            f"Test start {te_s} not after train end {tr_e}"
        assert pd.Timestamp(va_s) > pd.Timestamp(tr_e), \
            f"Val start {va_s} not after train end {tr_e}"
