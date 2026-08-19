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

import sys
from pathlib import Path

import pandas as pd

from forecasting import io

# --- Scoping decision (own this) ---------------------------------------------
CONFIG = {
    "category": "FOODS",   # one of: FOODS, HOBBIES, HOUSEHOLD
    "store": "CA_1",       # e.g. CA_1..CA_4, TX_1..TX_3, WI_1..WI_3
    "n_skus": 50,          # top-N SKUs by total units sold in this store/category
}

RAW = io.RAW
PROCESSED = io.PROCESSED

SALES_ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR_BASE_COLS = ["d", "date", "wm_yr_wk", "wday", "month", "year",
                      "event_name_1", "event_type_1"]


def _require_columns(df: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing expected column(s) {missing}. "
            "Is this the Kaggle m5-forecasting-accuracy file?"
        )


def main() -> None:
    io.ensure_processed(PROCESSED)

    sales_path = io.require_any_raw_file(io.SALES_CANDIDATES, RAW)
    calendar_path = io.require_raw_file(io.CALENDAR, RAW)
    io.require_raw_file(io.PRICES, RAW)

    print(f"Loading {sales_path.name} ...")
    sales = pd.read_csv(sales_path)
    _require_columns(sales, SALES_ID_COLS, sales_path)

    # Filter to one category + store.
    mask = (sales["cat_id"] == CONFIG["category"]) & (sales["store_id"] == CONFIG["store"])
    filtered = sales.loc[mask].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows for category={CONFIG['category']} store={CONFIG['store']}. "
            f"Available categories: {sorted(sales['cat_id'].unique())}; "
            f"stores: {sorted(sales['store_id'].unique())}."
        )
    sales = filtered

    # Rank SKUs by total units sold and keep the top N.
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    if not day_cols:
        raise ValueError(f"{sales_path} has no d_* day columns to melt.")
    units = sales[day_cols].apply(pd.to_numeric, errors="coerce")
    n_bad_units = int(units.isna().to_numpy().sum())
    if n_bad_units:
        raise ValueError(
            f"{n_bad_units} non-numeric unit value(s) in {sales_path.name}; "
            "refusing to build a parquet with missing sales."
        )
    sales = sales.assign(total_units=units.sum(axis=1))
    top = sales.nlargest(CONFIG["n_skus"], "total_units")
    kept_skus = top["item_id"].tolist()
    if len(kept_skus) < CONFIG["n_skus"]:
        print(
            f"WARNING: requested {CONFIG['n_skus']} SKUs but only {len(kept_skus)} exist "
            f"for category={CONFIG['category']} store={CONFIG['store']}.",
            file=sys.stderr,
        )
    print(f"Kept {len(kept_skus)} SKUs "
          f"(units sold range: {int(top['total_units'].min())}..{int(top['total_units'].max())})")

    # Wide -> long: one row per (item, day).
    long = top.melt(
        id_vars=SALES_ID_COLS,
        value_vars=day_cols,
        var_name="d",
        value_name="units",
    )
    long["units"] = pd.to_numeric(long["units"])

    # Join calendar for real dates + event/SNAP context.
    calendar = pd.read_csv(calendar_path)
    snap_col = f"snap_{CONFIG['store'][:2]}"  # snap_CA / snap_TX / snap_WI
    cal_cols = [*CALENDAR_BASE_COLS, snap_col]
    _require_columns(calendar, cal_cols, calendar_path)

    long = long.merge(calendar[cal_cols], on="d", how="left", validate="many_to_one")
    unmatched = long.loc[long["date"].isna(), "d"].unique()
    if len(unmatched) > 0:
        raise ValueError(
            f"{len(unmatched)} day column(s) have no row in {calendar_path.name} "
            f"(e.g. {list(unmatched[:5])}). The sales and calendar files are mismatched."
        )
    long["date"] = pd.to_datetime(long["date"], errors="raise")
    long = long.rename(columns={snap_col: "snap"})

    long = long.sort_values(["item_id", "date"]).reset_index(drop=True)

    sales_long, skus_txt = io.sales_long_path(PROCESSED), io.skus_path(PROCESSED)
    long.to_parquet(sales_long, index=False)
    skus_txt.write_text("\n".join(kept_skus), encoding="utf-8")

    print(f"\nWrote {sales_long}  ({len(long):,} rows)")
    print(f"Date range: {long['date'].min().date()} .. {long['date'].max().date()}")
    print(f"SKU list:   {skus_txt}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
