"""Subset the M5 dataset to a manageable slice and reshape to long format.

Step 1 of the build plan: scope the data. Reads the raw Kaggle M5 files from
data/raw/, filters to ONE category and ONE store, keeps the top-N SKUs by total
units sold (so we get series with enough signal to forecast), melts the wide
d_1..d_1913 columns into (sku, date, units) rows, and joins the calendar for
real dates plus SNAP/event flags.

Run:  python -m forecasting.prepare_data
Output: data/processed/sales_long.parquet  and  skus.txt

The category / store / SKU-count choices below are the scoping decision the
project doc says a human should own -- they change forecast difficulty and which
anomalies are visible. Edit CONFIG and re-run to re-scope.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Scoping decision (own this) ---------------------------------------------
CONFIG = {
    "category": "FOODS",   # one of: FOODS, HOBBIES, HOUSEHOLD
    "store": "CA_1",       # e.g. CA_1..CA_4, TX_1..TX_3, WI_1..WI_3
    "n_skus": 50,          # top-N SKUs by total units sold in this store/category
}

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

# Kaggle m5-forecasting-accuracy filenames. sales_train_evaluation.csv extends
# validation by 28 days (d_1914..d_1941); prefer it if present.
SALES_CANDIDATES = ["sales_train_evaluation.csv", "sales_train_validation.csv"]
CALENDAR = "calendar.csv"
PRICES = "sell_prices.csv"


def _find_sales_file() -> Path:
    for name in SALES_CANDIDATES:
        p = RAW / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of {SALES_CANDIDATES} found in {RAW}/. "
        "Download the M5 dataset from Kaggle (m5-forecasting-accuracy) and "
        "unzip the CSVs into data/raw/."
    )


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    sales_path = _find_sales_file()
    calendar_path = RAW / CALENDAR
    for p in (calendar_path, RAW / PRICES):
        if not p.exists():
            raise FileNotFoundError(f"Expected {p}; unzip all M5 CSVs into data/raw/.")

    print(f"Loading {sales_path.name} ...")
    sales = pd.read_csv(sales_path)

    # Filter to one category + store.
    mask = (sales["cat_id"] == CONFIG["category"]) & (sales["store_id"] == CONFIG["store"])
    sales = sales.loc[mask].copy()
    if sales.empty:
        raise ValueError(
            f"No rows for category={CONFIG['category']} store={CONFIG['store']}. "
            "Check the values against the dataset."
        )

    # Rank SKUs by total units sold and keep the top N.
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales["total_units"] = sales[day_cols].sum(axis=1)
    top = sales.nlargest(CONFIG["n_skus"], "total_units")
    kept_skus = top["item_id"].tolist()
    print(f"Kept {len(kept_skus)} SKUs "
          f"(units sold range: {int(top['total_units'].min())}..{int(top['total_units'].max())})")

    # Wide -> long: one row per (item, day).
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long = top.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="units",
    )

    # Join calendar for real dates + event/SNAP context.
    calendar = pd.read_csv(calendar_path)
    snap_col = f"snap_{CONFIG['store'][:2]}"  # snap_CA / snap_TX / snap_WI
    cal_cols = ["d", "date", "wm_yr_wk", "wday", "month", "year",
                "event_name_1", "event_type_1", snap_col]
    long = long.merge(calendar[cal_cols], on="d", how="left")
    long["date"] = pd.to_datetime(long["date"])
    long = long.rename(columns={snap_col: "snap"})

    long = long.sort_values(["item_id", "date"]).reset_index(drop=True)

    out = PROCESSED / "sales_long.parquet"
    long.to_parquet(out, index=False)
    (PROCESSED / "skus.txt").write_text("\n".join(kept_skus), encoding="utf-8")

    print(f"\nWrote {out}  ({len(long):,} rows)")
    print(f"Date range: {long['date'].min().date()} .. {long['date'].max().date()}")
    print(f"SKU list:   {PROCESSED / 'skus.txt'}")


if __name__ == "__main__":
    main()
