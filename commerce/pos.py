"""Mock POS (build plan step 7): replays real M5 sales + sell_prices.csv rows
for ONE product as a day-stepped feed -- not synthetic, so the "live" sales
the restocking agent reacts to are the exact same numbers the forecast model
trains and backtests on.

SKU (own this, per CLAUDE.md): FOODS_3_120. Chosen over SKUs already used
elsewhere in this project (e.g. FOODS_1_018, ~$1/unit) because its price is
flat at $4.98 across the whole history (zero price volatility -- clean
margin math, no promo-timing noise) and it has strong, steady volume
(~32 units/day, well above this scope's median) with the LightGBM model
beating seasonal-naive on it (49% vs 71% WAPE @ 28d) -- not one of the
problem SKUs flagged earlier.

The simulation clock runs over SIM_START_DATE..SIM_END_DATE, the exact same
held-out test window forecasting.lgbm_model never trains on -- so "the
commerce sim operates on genuinely unseen days" is true, not just asserted.
There's no real data past SIM_END_DATE (M5's history ends there), so that's
also the hard ceiling on how many simulated days exist.

This module is deliberately stateless -- a pure lookup over real data, no
mutable "current day" of its own. commerce/inventory.py owns the simulation
clock, since advancing a day is really an inventory-state transition (units
sold, stock arrived); pos.py just answers "what really happened on date X."
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from forecasting import io, lgbm_model
from forecasting.prepare_data import CONFIG as SCOPE_CONFIG

SKU = "FOODS_3_120"


@lru_cache(maxsize=1)
def _held_out_start() -> pd.Timestamp:
    wide = io.load_sales_wide()
    _, _, test_idxs = lgbm_model.time_split(len(wide.index))
    return wide.index[min(test_idxs)]


@lru_cache(maxsize=1)
def trading_days() -> pd.DatetimeIndex:
    wide = io.load_sales_wide()
    start = _held_out_start()
    return wide.index[(wide.index >= start)]


@lru_cache(maxsize=1)
def _load_prices() -> pd.DataFrame:
    """sell_prices.csv is weekly (wm_yr_wk), not daily; this broadcasts each
    week's price onto every simulated day in that week.

    Scoped to trading_days() rather than SKU's entire multi-year history --
    sell_prices.csv legitimately has no rows before an item was actually
    carried/priced at a store (FOODS_3_120 has no CA_1 price rows before
    week 11201, a real gap in the source data, not a bug), and the
    simulation never looks outside its own window anyway.

    Cached (@lru_cache): this reads the full 200MB+ sell_prices.csv, and
    get_day() calls it once per simulated day -- uncached, a 78-day
    simulation re-parses that file 78 times."""
    long_df = io.load_sales_long()
    days = trading_days()
    sku_dates = long_df.loc[
        (long_df["item_id"] == SKU) & (long_df["date"].isin(days)), ["date", "wm_yr_wk"]
    ].drop_duplicates()

    prices_path = io.require_raw_file(io.PRICES)
    prices = pd.read_csv(prices_path)
    prices = prices[(prices["item_id"] == SKU) & (prices["store_id"] == SCOPE_CONFIG["store"])]
    if prices.empty:
        raise ValueError(f"No sell_prices.csv rows for {SKU} at store {SCOPE_CONFIG['store']}.")

    merged = sku_dates.merge(prices[["wm_yr_wk", "sell_price"]], on="wm_yr_wk", how="left")
    missing = merged["date"][merged["sell_price"].isna()]
    if len(missing):
        raise ValueError(f"{len(missing)} simulated date(s) have no matching sell_prices.csv week for {SKU}.")
    return merged[["date", "sell_price"]].set_index("date").sort_index()


def get_day(date: pd.Timestamp | str) -> dict:
    """Real (units_sold, unit_price) for one simulated day. Raises if `date`
    is outside the simulated window -- there is either no real sales row
    (before the dataset starts) or no real row at all (past SIM_END_DATE)."""
    date = pd.Timestamp(date)
    days = trading_days()
    if date not in days:
        raise ValueError(
            f"{date.date()} is outside the simulated window "
            f"({days[0].date()}..{days[-1].date()}) for {SKU}."
        )

    wide = io.load_sales_wide()
    prices = _load_prices()
    return {
        "date": str(date.date()),
        "sku": SKU,
        "units_sold": int(wide.loc[date, SKU]),
        "unit_price": float(prices.loc[date, "sell_price"]),
    }


def average_sell_price() -> float:
    """Full-history average price for SKU -- used by commerce.suppliers to
    derive supplier offers as a discount/premium off a realistic retail price."""
    prices = _load_prices()
    return round(float(prices["sell_price"].mean()), 2)


if __name__ == "__main__":
    days = trading_days()
    print(f"SKU {SKU}: simulated window {days[0].date()}..{days[-1].date()} ({len(days)} days)")
    print(f"Average sell price: ${average_sell_price()}")
    print("First day:", get_day(days[0]))
    print("Last day: ", get_day(days[-1]))
