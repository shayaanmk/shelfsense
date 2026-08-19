"""Shared dataset locations and load/save helpers for the forecasting pipeline.

Every step reads from data/raw/ and writes to data/processed/; keeping the paths,
artifact names and the "did you unzip the M5 CSVs?" error in one place means a
re-scope or a directory move is a single edit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

# Kaggle m5-forecasting-accuracy filenames. sales_train_evaluation.csv extends
# validation by 28 days (d_1914..d_1941); prefer it if present.
SALES_CANDIDATES = ("sales_train_evaluation.csv", "sales_train_validation.csv")
CALENDAR = "calendar.csv"
PRICES = "sell_prices.csv"

SALES_LONG = PROCESSED / "sales_long.parquet"
SKUS_TXT = PROCESSED / "skus.txt"
BASELINE_METRICS = PROCESSED / "baseline_metrics.csv"

_DOWNLOAD_HINT = (
    "Download the M5 dataset from Kaggle (m5-forecasting-accuracy) and unzip "
    f"the CSVs into {RAW}/."
)


def ensure_processed() -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    return PROCESSED


def require_raw_file(name: str) -> Path:
    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}; {_DOWNLOAD_HINT}")
    return path


def require_any_raw_file(names: tuple[str, ...]) -> Path:
    """First of `names` present in data/raw/, for files with alternative releases."""
    for name in names:
        path = RAW / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {list(names)} found in {RAW}/. {_DOWNLOAD_HINT}")


def load_sales_long() -> pd.DataFrame:
    if not SALES_LONG.exists():
        raise FileNotFoundError(
            f"Expected {SALES_LONG}; run `python -m forecasting.prepare_data` first."
        )
    return pd.read_parquet(SALES_LONG)


def load_sales_wide(values: str = "units") -> pd.DataFrame:
    """Sales as a date x item_id matrix, the shape the forecasters consume."""
    return load_sales_long().pivot(index="date", columns="item_id", values=values).sort_index()
