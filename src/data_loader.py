"""
Load raw data from locally downloaded WRDS CSV files and convert to parquet.
Run once: cd /Users/alexchen/Downloads/xx && PYTHONPATH=src python3 -m src.data_loader
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_RAW, START_DATE, END_DATE

DATA_RAW.mkdir(parents=True, exist_ok=True)
DOWNLOADS = DATA_RAW / "downloads"


def _to_datetime_safe(series: pd.Series, format: str = None) -> pd.Series:
    """
    pd.to_datetime crashes on Python 3.14 with certain inputs.
    Workaround: parse via year/month/day arithmetic for YYYYMMDD integers,
    or use dateutil for YYYY-MM-DD strings.
    """
    s = series.astype(str).str.strip()
    # YYYYMMDD integer format (FF factors)
    if format == "%Y%m%d":
        numeric = pd.to_numeric(s, errors="coerce").dropna()
        d = numeric.astype(int)
        return pd.to_datetime({
            "year":  d // 10000,
            "month": (d % 10000) // 100,
            "day":   d % 100,
        }, errors="coerce").reindex(series.index)
    # YYYY-MM-DD string format (CRSP)
    return pd.Series(
        pd.array([pd.Timestamp(v) if len(v) == 10 else pd.NaT for v in s]),
        index=series.index,
    )


def load_crsp_daily() -> pd.DataFrame:
    """
    crsp_dsf.csv — columns: PERMNO, date, EXCHCD, SICCD, PRC, VOL, RET, SHROUT, CFACPR
    """
    f = DOWNLOADS / "crsp_dsf.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")

    print("  Reading crsp_dsf.csv (2 GB — may take a few minutes)...")
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    df["date"]   = _to_datetime_safe(df["date"])
    df["prc"]    = pd.to_numeric(df["prc"],    errors="coerce").abs()
    df["ret"]    = pd.to_numeric(df["ret"],    errors="coerce")
    df["vol"]    = pd.to_numeric(df["vol"],    errors="coerce").abs()
    df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce")
    df["cfacpr"] = pd.to_numeric(df["cfacpr"], errors="coerce")
    df["exchcd"] = pd.to_numeric(df["exchcd"], errors="coerce")
    df["siccd"]  = pd.to_numeric(df["siccd"],  errors="coerce")

    df["mktcap"]    = df["prc"] * df["shrout"] * 1000   # shrout in thousands
    df["dollar_vol"] = df["prc"] * df["vol"]

    df = df[df["date"].between(START_DATE, END_DATE)]
    df = df.dropna(subset=["permno", "date"])
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    return df


def load_crsp_delist() -> pd.DataFrame:
    """
    crsp_delist.csv — full daily file with only PERMNO, date, DLSTCD, DLRET.
    We filter to rows where dlstcd is not null (actual delisting events).
    """
    f = DOWNLOADS / "crsp_delist.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")

    print("  Reading crsp_delist.csv...")
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    df["date"]   = _to_datetime_safe(df["date"])
    df["dlret"]  = pd.to_numeric(df["dlret"],  errors="coerce")
    df["dlstcd"] = pd.to_numeric(df["dlstcd"], errors="coerce")

    # Keep only actual delisting events
    df = df[df["dlstcd"].notna()].copy()
    df = df.rename(columns={"date": "dlstdt"})
    df = df[["permno", "dlstdt", "dlret", "dlstcd"]]
    df = df.sort_values(["permno", "dlstdt"]).reset_index(drop=True)
    return df


def load_crsp_names() -> pd.DataFrame:
    """
    crsp_names.csv — Stock Header Info.
    Columns: PERMNO, HEXCD, HSICCD, HTICK, BEGDAT, ENDDAT
    """
    f = DOWNLOADS / "crsp_names.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")

    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    # Rename header-level columns to standard names
    rename = {
        "hexcd":   "exchcd",
        "hsiccd":  "siccd",
        "htick":   "ticker",
        "begdat":  "namedt",
        "enddat":  "nameendt",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in ["namedt", "nameendt"]:
        if col in df.columns:
            df[col] = _to_datetime_safe(df[col])

    return df


def load_compustat_quarterly() -> pd.DataFrame:
    """
    compustat_fundq.csv — Compustat Fundamentals Quarterly.
    gvkey and datadate are auto-included; rdq and financials are user-selected.
    """
    f = DOWNLOADS / "compustat_fundq.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")

    print("  Reading compustat_fundq.csv...")
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    for col in ["datadate", "rdq"]:
        if col in df.columns:
            df[col] = _to_datetime_safe(df[col])

    # Standard Compustat filters (columns may already be filtered by WRDS)
    for col, val in [("indfmt","INDL"),("datafmt","STD"),("popsrc","D"),("consol","C")]:
        if col in df.columns:
            df = df[df[col] == val]

    df = df.dropna(subset=["rdq"])
    df = df.sort_values(["gvkey", "rdq"]).reset_index(drop=True)
    return df


def load_crsp_compustat_link() -> pd.DataFrame:
    """
    crsp_compustat_link.csv — CCM link table.
    Columns: gvkey, lpermno, linktype, linkprim, linkdt, linkenddt
    """
    f = DOWNLOADS / "crsp_compustat_link.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")

    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    if "lpermno" in df.columns and "permno" not in df.columns:
        df = df.rename(columns={"lpermno": "permno"})

    for col in ["linkdt", "linkenddt"]:
        if col in df.columns:
            df[col] = _to_datetime_safe(df[col])

    df = df[df["linktype"].isin(["LU", "LC"])]
    df = df[df["linkprim"].isin(["P", "C"])]
    return df


def load_fama_french() -> pd.DataFrame:
    """
    ff_factors_daily.csv — Ken French daily factors.
    Header rows + copyright footer handled automatically.
    """
    f = DOWNLOADS / "ff_factors_daily.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")

    with open(f) as fh:
        lines = fh.readlines()

    skip = next(i for i, l in enumerate(lines) if "Mkt-RF" in l or "MKT-RF" in l.upper())

    df = pd.read_csv(f, skiprows=skip, header=0,
                     names=["date", "mktrf", "smb", "hml", "rf"],
                     on_bad_lines="skip")

    # Drop non-numeric rows (copyright line etc.)
    df = df[pd.to_numeric(df["date"], errors="coerce").notna()].copy()

    df["date"] = _to_datetime_safe(df["date"], format="%Y%m%d")
    for col in ["mktrf", "smb", "hml", "rf"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100

    df = df.dropna(subset=["date"])
    df = df[df["date"].between(START_DATE, END_DATE)]
    return df.sort_values("date").reset_index(drop=True)


def check_downloads() -> bool:
    files = {
        "crsp_dsf.csv":             DOWNLOADS / "crsp_dsf.csv",
        "crsp_delist.csv":          DOWNLOADS / "crsp_delist.csv",
        "crsp_names.csv":           DOWNLOADS / "crsp_names.csv",
        "compustat_fundq.csv":      DOWNLOADS / "compustat_fundq.csv",
        "crsp_compustat_link.csv":  DOWNLOADS / "crsp_compustat_link.csv",
        "ff_factors_daily.csv":     DOWNLOADS / "ff_factors_daily.csv",
    }
    print("\n── Download status ──────────────────────────────")
    all_ok = True
    for name, path in files.items():
        if path.exists():
            size = path.stat().st_size / 1e6
            print(f"  ✓ {name:<35} {size:>7.1f} MB")
        else:
            print(f"  ✗ {name:<35} MISSING")
            all_ok = False
    print("─────────────────────────────────────────────────")
    return all_ok


def main():
    if not check_downloads():
        print("\nSome files missing. Add them to data/raw/downloads/ and re-run.")
        return

    print("\nConverting files to parquet...")

    print("Fama-French factors...")
    ff = load_fama_french()
    ff.to_parquet(DATA_RAW / "ff_factors_daily.parquet", index=False)
    print(f"  ✓ {len(ff):,} rows | {ff.date.min().date()} to {ff.date.max().date()}")

    print("CRSP stock header (names)...")
    names = load_crsp_names()
    names.to_parquet(DATA_RAW / "crsp_names.parquet", index=False)
    print(f"  ✓ {len(names):,} rows")

    print("CRSP-Compustat link table...")
    link = load_crsp_compustat_link()
    link.to_parquet(DATA_RAW / "crsp_compustat_link.parquet", index=False)
    print(f"  ✓ {len(link):,} rows")

    print("Compustat quarterly fundamentals...")
    fundq = load_compustat_quarterly()
    fundq.to_parquet(DATA_RAW / "compustat_fundq.parquet", index=False)
    print(f"  ✓ {len(fundq):,} rows")

    print("CRSP delisting returns...")
    delist = load_crsp_delist()
    delist.to_parquet(DATA_RAW / "crsp_delist.parquet", index=False)
    print(f"  ✓ {len(delist):,} delisting events")

    print("CRSP daily stock file (largest — ~5 min)...")
    crsp = load_crsp_daily()
    crsp.to_parquet(DATA_RAW / "crsp_dsf.parquet", index=False)
    print(f"  ✓ {len(crsp):,} rows | {crsp.date.min().date()} to {crsp.date.max().date()}")

    print("\nAll done. Run universe.py next.")


if __name__ == "__main__":
    main()
