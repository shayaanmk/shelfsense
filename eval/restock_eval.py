"""Restocking-agent vs. naive-baseline simulated-profit comparison (build
plan step 8's definition of done: "beats a naive always-reorder-to-target
baseline on simulated profit").

Runs TWO independent 78-day simulations over commerce.pos's real replay
window -- one driven by agent.restock_agent's newsvendor decision (per-day
profit-optimal order quantity AND supplier choice, full guardrails), one by
a naive policy: same reorder trigger and target level (so both react at the
same moment), always orders from supplier_a (the cheaper one -- the obvious
non-optimizing choice) with no profit optimization, no spend cap, no
large-order queueing. Both keep a basic "don't reorder while one's already
in transit" check, so the comparison isolates economic decision quality
rather than just which one avoids trivially duplicating orders.

Both runs replay the IDENTICAL real demand (commerce.pos has no randomness),
scored with the same profit formula:
  profit = revenue (fulfilled sales x sell price)
         - cost of goods received (from the PO ledger)
         - holding cost (accrued daily on ending on-hand, CLAUDE.md's $15/yr)
         - stockout cost ($45/unit x unmet demand, CLAUDE.md's Cu)

Reuses the live commerce.inventory/commerce.orders state files (clearing
them before each run) -- not meant to run concurrently with anything else
touching that state. Ends with inventory.reset() so it doesn't leave the
demo state mid-simulation.

Run:  python -m eval.restock_eval
Output: data/processed/eval_restock_comparison.json
"""

from __future__ import annotations

import json
import os
from typing import Callable

import pandas as pd

from agent import restock_agent
from commerce import inventory, orders, pos, suppliers
from forecasting import io

NAIVE_SUPPLIER = "supplier_a"


def _naive_decision(sku: str) -> dict:
    state = inventory.load_state()
    as_of = pd.Timestamp(state["current_date"])

    if not orders.pending_orders(sku).empty:
        return {"action": "no_order", "reason": "order already pending"}
    if state["on_hand"] > state["reorder_point"]:
        return {"action": "no_order", "reason": "above reorder point"}

    offer = suppliers.get_offer(NAIVE_SUPPLIER)
    qty = max(state["target_level"] - state["on_hand"], offer["moq"])
    po = orders.place_order(
        sku,
        NAIVE_SUPPLIER,
        round(qty),
        offer["unit_price"],
        str(as_of.date()),
        offer["lead_time_days"],
        rationale="naive baseline: always reorder to target level, fixed cheaper supplier, no profit optimization.",
    )
    return {"action": "order", "po_id": po.po_id, "quantity": qty}


def _run_policy(decide_fn: Callable[[], dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists(restock_agent.DECISIONS_PATH):
        os.remove(restock_agent.DECISIONS_PATH)
    if os.path.exists(orders.LEDGER_PATH):
        os.remove(orders.LEDGER_PATH)
    inventory.reset()

    for _ in range(len(pos.trading_days())):
        decide_fn()
        inventory.advance_day()

    log = inventory.load_log()
    received = orders.all_orders(pos.SKU)
    return log, received[received["status"] == "received"]


def _simulated_profit(log: pd.DataFrame, received_orders: pd.DataFrame) -> dict:
    revenue = float((log["fulfilled"] * log["unit_price"]).sum())
    cogs = float(received_orders["total_cost"].sum())
    holding_cost = float((log["on_hand_end_of_day"] / 365 * suppliers.HOLDING_COST_PER_UNIT_PER_YEAR).sum())
    stockout_cost = float(log["unmet_demand"].sum()) * suppliers.STOCKOUT_COST_PER_UNIT
    profit = revenue - cogs - holding_cost - stockout_cost
    return {
        "revenue": round(revenue, 2),
        "cogs": round(cogs, 2),
        "holding_cost": round(holding_cost, 2),
        "stockout_cost": round(stockout_cost, 2),
        "profit": round(profit, 2),
        "stockout_days": int(log["stockout"].sum()),
        "total_unmet_demand": round(float(log["unmet_demand"].sum()), 1),
        "n_orders_received": len(received_orders),
    }


def main() -> None:
    print("Running agent.restock_agent policy...")
    agent_log, agent_received = _run_policy(lambda: restock_agent.run())
    agent_result = _simulated_profit(agent_log, agent_received)

    print("Running naive always-reorder-to-target baseline...")
    naive_log, naive_received = _run_policy(lambda: _naive_decision(pos.SKU))
    naive_result = _simulated_profit(naive_log, naive_received)

    inventory.reset()  # leave the demo state clean, not mid-comparison

    result = {"restock_agent": agent_result, "naive_baseline": naive_result}
    out = io.PROCESSED / "eval_restock_comparison.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}\n")

    print(f"{'metric':<22}{'restock_agent':>16}{'naive_baseline':>16}")
    for key in (
        "revenue",
        "cogs",
        "holding_cost",
        "stockout_cost",
        "profit",
        "stockout_days",
        "total_unmet_demand",
        "n_orders_received",
    ):
        print(f"{key:<22}{agent_result[key]:>16}{naive_result[key]:>16}")

    improvement = agent_result["profit"] - naive_result["profit"]
    print(f"\nrestock_agent profit - naive_baseline profit = {improvement:+.2f}")


if __name__ == "__main__":
    main()
