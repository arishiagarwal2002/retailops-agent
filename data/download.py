"""Download UCI Online Retail II and cache it as Parquet.

The source is a ~45 MB .xlsx with two sheets (2009-2010 and 2010-2011).
Parsing it with openpyxl takes a few minutes, so we do it once and keep a
Parquet copy for every later step.

    python -m data.download
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = Path(__file__).parent / "raw"
XLSX = RAW_DIR / "online_retail_II.xlsx"
PARQUET = RAW_DIR / "online_retail_ii.parquet"


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if XLSX.exists():
        print(f"already downloaded: {XLSX}")
        return XLSX
    print(f"downloading {URL}")
    resp = requests.get(URL, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(RAW_DIR)
    return XLSX


def to_parquet() -> Path:
    if PARQUET.exists():
        print(f"already converted: {PARQUET}")
        return PARQUET
    print("reading both sheets (this takes a few minutes)...")
    sheets = pd.read_excel(XLSX, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    df["Description"] = df["Description"].astype("string")
    df.to_parquet(PARQUET, index=False)
    print(f"wrote {len(df):,} rows -> {PARQUET}")
    return PARQUET


if __name__ == "__main__":
    download()
    to_parquet()
