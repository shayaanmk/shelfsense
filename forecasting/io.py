"""Shared dataset locations and load/save helpers for the forecasting pipeline.

Every step reads from data/raw/ and writes to data/processed/; keeping the paths,
artifact names, the "did you unzip the M5 CSVs?" error and the processed-data
validation in one place means a re-scope or a directory move is a single edit.

The directory arguments default to RAW / PROCESSED and exist so a caller (or a
test) can point a run at a different tree.
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

SALES_LONG_NAME = "sales_long.parquet"
SKUS_NAME = "skus.txt"
BASELINE_METRICS_NAME = "baseline_metrics.csv"

_DOWNLOAD_HINT = (
    "Download the M5 dataset from Kaggle (m5-forecasting-accuracy) and unzip "
    "the CSVs into data/raw/."
)


def ensure_processed(processed: Path | None = None) -> Path:
    processed = PROCESSED if processed is None else processed
    processed.mkdir(parents=True, exist_ok=True)
    return processed


def require_raw_file(name: str, raw: Path | None = None) -> Path:
    path = (RAW if raw is None else raw) / name
    if not path.exists():
        raise FileNotFoundError(f"Expected {path}; {_DOWNLOAD_HINT}")
    return path


def require_any_raw_file(names: tuple[str, ...], raw: Path | None = None) -> Path:
    """First of `names` present in the raw dir, for files with alternative releases."""
    raw = RAW if raw is None else raw
    for name in names:
        path = raw / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {list(names)} found in {raw}/. {_DOWNLOAD_HINT}")


def sales_long_path(processed: Path | None = None) -> Path:
    return (PROCESSED if processed is None else processed) / SALES_LONG_NAME


def skus_path(processed: Path | None = None) -> Path:
    return (PROCESSED if processed is None else processed) / SKUS_NAME


def baseline_metrics_path(processed: Path | None = None) -> Path:
    return (PROCESSED if processed is None else processed) / BASELINE_METRICS_NAME


def load_sales_long(processed: Path | None = None, values: str = "units") -> pd.DataFrame:
    path = sales_long_path(processed)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m forecasting.prepare_data` first."
        )
    long_df = pd.read_parquet(path)
    missing = [c for c in ("date", "item_id", values) if c not in long_df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}.")
    return long_df


def load_sales_wide(values: str = "units", processed: Path | None = None) -> pd.DataFrame:
    """Sales as a date x item_id matrix, the shape the forecasters consume."""
    long_df = load_sales_long(processed, values)
    try:
        return long_df.pivot(index="date", columns="item_id", values=values).sort_index()
    except ValueError as exc:
        raise ValueError(
            f"Cannot reshape {sales_long_path(processed)} to one row per date: duplicate "
            f"(date, item_id) pairs. ({exc})"
        ) from exc
