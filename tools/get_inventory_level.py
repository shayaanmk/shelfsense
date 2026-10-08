"""get_inventory_level(sku, as_of): current (or historical) simulated
inventory state for one of the agent's tools -- reads
data/processed/inventory.parquet, produced by tools/inventory_sim.py.
"""

from __future__ import annotations

import pandas as pd

from forecasting import io


def get_inventory_level(sku: str, as_of: str | None = None) -> dict:
    inv = io.load_inventory()
    item_inv = inv[inv["item_id"] == sku].sort_values("date")
    if item_inv.empty:
        raise ValueError(f"Unknown SKU {sku!r}; no inventory simulation for it.")

    if as_of is None:
        row = item_inv.iloc[-1]
    else:
        as_of_ts = pd.Timestamp(as_of)
        matches = item_inv[item_inv["date"] == as_of_ts]
        if matches.empty:
            lo, hi = item_inv["date"].min().date(), item_inv["date"].max().date()
            raise ValueError(f"No simulated inventory for {sku} on {as_of}; range is {lo}..{hi}.")
        row = matches.iloc[0]

    trailing = item_inv[item_inv["date"] <= row["date"]].tail(28)
    avg_daily_demand = float(trailing["demand"].mean())
    days_of_supply = round(row["on_hand"] / avg_daily_demand, 1) if avg_daily_demand > 0 else None

    restocks = item_inv[(item_inv["date"] <= row["date"]) & (item_inv["order_arrival_qty"] > 0)]
    last_restock_date = str(restocks["date"].max().date()) if not restocks.empty else None

    return {
        "sku": sku,
        "as_of_date": str(row["date"].date()),
        "on_hand": round(float(row["on_hand"]), 1),
        "days_of_supply": days_of_supply,
        "reorder_point": round(float(row["reorder_point"]), 1),
        "target_level": round(float(row["target_level"]), 1),
        "stockout_today": bool(row["stockout"]),
        "last_restock_date": last_restock_date,
    }
