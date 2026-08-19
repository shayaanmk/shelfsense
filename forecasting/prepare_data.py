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

import pandas as pd

from forecasting import io

# --- Scoping decision (own this) ---------------------------------------------
CONFIG = {
    "category": "FOODS",   # one of: FOODS, HOBBIES, HOUSEHOLD
    "store": "CA_1",       # e.g. CA_1..CA_4, TX_1..TX_3, WI_1..WI_3
    "n_skus": 50,          # top-N SKUs by total units sold in this store/category
}


def main() -> None:
    io.ensure_processed()

    sales_path = io.require_any_raw_file(io.SALES_CANDIDATES)
    calendar_path = io.require_raw_file(io.CALENDAR)
    io.require_raw_file(io.PRICES)

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

    long.to_parquet(io.SALES_LONG, index=False)
    io.SKUS_TXT.write_text("\n".join(kept_skus), encoding="utf-8")

    print(f"\nWrote {io.SALES_LONG}  ({len(long):,} rows)")
    print(f"Date range: {long['date'].min().date()} .. {long['date'].max().date()}")
    print(f"SKU list:   {io.SKUS_TXT}")


if __name__ == "__main__":
    main()
