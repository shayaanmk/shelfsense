"""Live on-hand inventory for commerce.pos.SKU (build plan step 7).

Owns the simulation clock (current_date) as well as on-hand units -- "what
day is it" and "how much stock do we have" are really one piece of state
that advances together one day at a time, so splitting them across two
files would just mean keeping them in sync by hand.

State persists to data/commerce/inventory_state.json so a scheduled agent
(step 8) picks up where the last run left off instead of resetting each
invocation. Call reset() to start the simulation over from SIM_START_DATE.

Initial on-hand and the reorder point/target level are seeded from the SAME
(s, S) reorder-point policy tools/inventory_sim.py already uses for the
50-SKU training data (tools.inventory_sim.reorder_policy) -- one policy
definition, reused, rather than inventing a second one for the single
commerce SKU.
"""

from __future__ import annotations

import json

import pandas as pd

from commerce import orders, pos
from forecasting import io
from tools.inventory_sim import reorder_policy

STATE_PATH = io.PROCESSED.parent / "commerce" / "inventory_state.json"
LOG_PATH = io.PROCESSED.parent / "commerce" / "commerce_log.csv"

LOG_COLUMNS = [
    "date",
    "units_sold",
    "unit_price",
    "fulfilled",
    "unmet_demand",
    "stockout",
    "units_received",
    "on_hand_end_of_day",
]


def _policy() -> tuple[float, float]:
    """(s, S) fit on this SKU's real demand over its whole simulated window --
    the same held-out slice pos.trading_days() replays, so the policy reflects
    this SKU's actual recent volume, not stale history from years earlier."""
    wide = io.load_sales_wide()
    days = pos.trading_days()
    demand = wide.loc[days, pos.SKU].to_numpy(dtype=float)
    return reorder_policy(demand)


def _initial_state() -> dict:
    days = pos.trading_days()
    s, target = _policy()
    return {
        "sku": pos.SKU,
        "current_date": str(days[0].date()),
        "on_hand": target,
        "reorder_point": s,
        "target_level": target,
    }


def load_state() -> dict:
    if not STATE_PATH.exists():
        reset()
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def reset() -> dict:
    """Restarts the simulation clock at SIM_START_DATE with a fully-stocked
    (on_hand = target_level) inventory, and truncates the day-by-day log --
    does NOT touch the PO ledger (commerce.orders owns that separately)."""
    state = _initial_state()
    _save_state(state)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False)
    return state


def get_state() -> dict:
    return load_state()


def is_finished() -> bool:
    state = load_state()
    days = pos.trading_days()
    return pd.Timestamp(state["current_date"]) > days[-1]


def advance_day() -> dict:
    """Steps the simulation forward exactly one day: receives any orders due
    today, sells today's real units (clamped at on-hand -- unmet demand is
    lost, not backordered, matching tools/inventory_sim.py's convention),
    logs the day, and moves the clock to tomorrow. Raises once the
    simulated window (bounded by real data) is exhausted."""
    state = load_state()
    days = pos.trading_days()
    current_date = pd.Timestamp(state["current_date"])
    if current_date > days[-1]:
        raise ValueError(
            f"Simulation already finished at {days[-1].date()}; call reset() to run it again."
        )

    today = pos.get_day(current_date)

    received = orders.receive_due_orders(str(current_date.date()))
    units_received = sum(po.quantity for po in received)
    on_hand = state["on_hand"] + units_received

    units_sold = today["units_sold"]
    fulfilled = min(on_hand, units_sold)
    unmet = units_sold - fulfilled
    on_hand -= fulfilled
    stockout = on_hand <= 0 and units_sold > 0

    state["on_hand"] = on_hand
    state["current_date"] = str((current_date + pd.Timedelta(days=1)).date())
    _save_state(state)

    log_row = {
        "date": str(current_date.date()),
        "units_sold": units_sold,
        "unit_price": today["unit_price"],
        "fulfilled": fulfilled,
        "unmet_demand": unmet,
        "stockout": stockout,
        "units_received": units_received,
        "on_hand_end_of_day": on_hand,
    }
    log = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.concat([log, pd.DataFrame([log_row])], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)

    return log_row


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_PATH, parse_dates=["date"])


if __name__ == "__main__":
    state = reset()
    print(f"Reset: {state}\n")
    for _ in range(5):
        print(advance_day())
    print("\nState after 5 days:", load_state())
