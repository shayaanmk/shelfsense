"""Reorder-point (s, S) inventory simulation, run once over real M5 sales
history to produce a synthetic-but-mechanistic daily inventory series.

M5 has no real inventory data, so get_inventory_level() and the supply-side
half of detect_anomalies() need something to read. Rather than fabricate
numbers, this simulates a standard continuous-review (s, S) policy driven by
actual daily units sold: demand depletes on-hand stock; when the inventory
position (on-hand + outstanding orders) drops to the reorder point s, an
order for (S - position) is placed and arrives after a fixed lead time. A
stockout is on-hand hitting 0 while there was still unmet demand that day --
a real, explainable mechanism, not a random flag.

Policy parameters (own these, per CLAUDE.md):
  - LEAD_TIME_DAYS = 3: days between placing and receiving an order.
  - SERVICE_Z = 1.65: safety-stock z-score, ~95% cycle service level.
  - CYCLE_DAYS = 7: the order-up-to level covers this many extra days of
    average demand beyond the reorder point (roughly weekly replenishment).
  - s and S are fit per SKU from that SKU's full-history mean/std of daily
    units -- a real planner would set policy from historical demand stats too.

Run:  python -m tools.inventory_sim
Output: data/processed/inventory.parquet
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from forecasting import io

LEAD_TIME_DAYS = 3
SERVICE_Z = 1.65
CYCLE_DAYS = 7


def reorder_policy(units: np.ndarray) -> tuple[float, float]:
    mean = float(units.mean())
    std = float(units.std())
    safety_stock = SERVICE_Z * std * np.sqrt(LEAD_TIME_DAYS)
    s = mean * LEAD_TIME_DAYS + safety_stock
    target = s + mean * CYCLE_DAYS
    return s, target


def simulate_item(
    dates: pd.DatetimeIndex,
    demand: np.ndarray,
    s: float,
    target: float,
    delay_orders_on: set[int] | None = None,
    extra_lead_days: int = 0,
) -> pd.DataFrame:
    """`delay_orders_on`/`extra_lead_days` let eval.anomaly_eval inject a
    "supply delay" scenario: any order placed on one of those day-indices
    arrives `extra_lead_days` later than LEAD_TIME_DAYS -- e.g. a shipment
    delay -- rather than editing on_hand directly, so a resulting stockout
    is a real consequence of the delay, not a fabricated flag. Defaults
    reproduce the undelayed policy exactly."""
    n = len(demand)
    on_hand = np.zeros(n)
    fulfilled = np.zeros(n)
    unmet = np.zeros(n)
    stockout = np.zeros(n, dtype=bool)
    order_placed_qty = np.zeros(n)
    order_arrival_qty = np.zeros(n)

    pending: list[tuple[int, float]] = []  # (arrival_index, qty)
    stock = target  # start fully stocked

    for t in range(n):
        arriving = sum(qty for idx, qty in pending if idx == t)
        if arriving:
            stock += arriving
            order_arrival_qty[t] = arriving
            pending = [(idx, qty) for idx, qty in pending if idx != t]

        d = demand[t]
        served = min(stock, d)
        stock -= served
        fulfilled[t] = served
        unmet[t] = d - served
        stockout[t] = stock <= 0 and d > 0
        on_hand[t] = stock

        position = stock + sum(qty for _, qty in pending)
        if position <= s:
            qty = target - position
            lead = LEAD_TIME_DAYS + (extra_lead_days if delay_orders_on and t in delay_orders_on else 0)
            pending.append((t + lead, qty))
            order_placed_qty[t] = qty

    return pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
            "on_hand": on_hand,
            "fulfilled": fulfilled,
            "unmet_demand": unmet,
            "stockout": stockout,
            "order_placed_qty": order_placed_qty,
            "order_arrival_qty": order_arrival_qty,
        }
    )


def main() -> None:
    wide = io.load_sales_wide()
    dates = wide.index

    frames = []
    for item in wide.columns:
        demand = wide[item].to_numpy(dtype=float)
        s, target = reorder_policy(demand)
        sim = simulate_item(dates, demand, s, target)
        sim.insert(0, "item_id", item)
        sim["reorder_point"] = s
        sim["target_level"] = target
        frames.append(sim)

    result = pd.concat(frames, ignore_index=True)

    io.ensure_processed()
    out = io.inventory_path()
    result.to_parquet(out, index=False)

    n_stockout_days = int(result["stockout"].sum())
    n_items_stocked_out = result.loc[result["stockout"], "item_id"].nunique()
    print(f"Wrote {out}  ({len(result):,} rows)")
    print(f"Stockout days: {n_stockout_days:,} across {n_items_stocked_out}/{wide.shape[1]} SKUs")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
