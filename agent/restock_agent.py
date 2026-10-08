"""Autonomous restocking agent (build plan step 8) -- schedule-triggered,
no human in the loop before it fires, as distinct from agent/copilot.py's
request/response pattern. Decides order quantity + supplier via a
newsvendor profit calculation, bounded by guardrails CLAUDE.md already
decided; every decision -- including "do nothing" -- logs its full inputs
and rationale to data/commerce/restock_decisions.csv, since nothing reviews
it before it acts.

## The decision (newsvendor, per CLAUDE.md's already-decided cost inputs)

For each supplier, estimate demand over ITS lead time as Normal(mu, sigma):
  - mu from the trained forecast (tools.get_forecast), summed over the
    lead time.
  - sigma from this SKU's own trailing-180-day daily volatility, scaled by
    sqrt(lead_time_days) under an iid-days approximation -- a real distinct
    estimate per supplier, since a 10-day lead time (Supplier A) carries
    more forecast uncertainty than a 3-day one (Supplier B).
Then pick the order-up-to level S* at the critical fractile
Cu/(Cu+Co) (commerce.suppliers.critical_ratio -- Cu=$45 stockout cost,
Co=holding cost prorated to that supplier's lead time), compute the
resulting order quantity (respecting MOQ), and its expected profit via the
standard newsvendor closed form. Suppliers are compared on expected profit
PER DAY (profit / lead_time_days), not raw expected profit -- a 10-day
order and a 3-day order cover different amounts of demand, so comparing
them unnormalized would structurally favor whichever supplier has the
longer lead time regardless of actual profitability. A cheaper-but-slower
supplier can still lose to a pricier-but-faster one once its lead time's
extra demand risk and holding cost are priced in, not just compared on
unit cost.

## Guardrails (CLAUDE.md's exact specs, not reinvented here)

  - Spend cap: an order can't exceed 50% of a trailing-180-day supply-cost
    basis. Cold start (no 6 months of real PO history yet -- true for this
    entire 78-day demo, by construction): basis = trailing_180d real M5
    units x supplier price, per CLAUDE.md.
  - Duplicate-order guard: a candidate order within ~5% of an already-
    pending (in-transit) order for the same (sku, supplier) is silently
    skipped -- logged as a detected duplicate, not placed.
  - Large-order guard: an order >70% of the spend cap is "large"; if a
    second large order for this SKU would land within one lead-time window
    of an existing one, this one is NOT auto-placed -- it's recorded with
    status="pending_approval" (commerce.orders) instead.
  - Idempotency: re-running for a date already logged in
    restock_decisions.csv returns that logged decision rather than
    re-deciding (and potentially double-ordering) for the same day.
  - dry_run: compute and log the decision, but skip the actual
    commerce.orders.place_order call.

Run:  python -m agent.restock_agent               (decide for today, live)
      python -m agent.restock_agent --dry-run      (decide, don't place)
"""

from __future__ import annotations

import sys

import pandas as pd
from scipy.stats import norm

from commerce import inventory, orders, pos, suppliers
from forecasting import io
from tools.get_forecast import get_forecast

DECISIONS_PATH = io.PROCESSED.parent / "commerce" / "restock_decisions.csv"
DECISION_COLUMNS = [
    "date",
    "sku",
    "action",
    "supplier_id",
    "quantity",
    "unit_price",
    "order_value",
    "expected_profit",
    "expected_profit_per_day",
    "po_id",
    "rationale",
]

SUPPLIER_IDS = ("supplier_a", "supplier_b")
SPEND_CAP_FRACTION = 0.50
LARGE_ORDER_FRACTION = 0.70
DUPLICATE_QTY_TOLERANCE = 0.05
TRAILING_COST_WINDOW_DAYS = 180
VOLATILITY_LOOKBACK_DAYS = 180
REAL_PO_HISTORY_MIN_DAYS = 180  # "6 months" cold-start -> real-ledger switch, per CLAUDE.md


def _normal_loss(z: float) -> float:
    """E[(Z-z)+] for standard normal Z -- the newsvendor unit loss function."""
    return norm.pdf(z) - z * (1 - norm.cdf(z))


def _expected_profit(order_up_to: float, mu: float, sigma: float, margin: float, co: float, cu: float) -> float:
    if sigma <= 0:
        shortfall = max(mu - order_up_to, 0.0)
        overage = max(order_up_to - mu, 0.0)
        sold = min(order_up_to, mu)
        return margin * sold - co * overage - cu * shortfall
    z = (order_up_to - mu) / sigma
    expected_shortfall = sigma * _normal_loss(z)
    return margin * mu - co * (order_up_to - mu) - (margin + co + cu) * expected_shortfall


def _lead_time_demand_stats(sku: str, lead_time_days: int, as_of_date: pd.Timestamp) -> tuple[float, float]:
    forecast = get_forecast(sku, lead_time_days, as_of=as_of_date)
    mu = sum(day["predicted_units"] for day in forecast["forecast"])

    wide = io.load_sales_wide()
    history = wide.loc[wide.index < as_of_date, sku].tail(VOLATILITY_LOOKBACK_DAYS)
    daily_std = float(history.std()) if len(history) > 1 else 0.0
    sigma = daily_std * (lead_time_days**0.5)
    return mu, sigma


def _evaluate_supplier(sku: str, supplier_id: str, as_of_date: pd.Timestamp, inventory_position: float) -> dict:
    offer = suppliers.get_offer(supplier_id)
    margin = suppliers.margin_per_unit(supplier_id)
    co = suppliers.HOLDING_COST_PER_UNIT_PER_YEAR * (offer["lead_time_days"] / 365)
    cu = suppliers.STOCKOUT_COST_PER_UNIT
    ratio = suppliers.critical_ratio(offer["lead_time_days"])

    mu, sigma = _lead_time_demand_stats(sku, offer["lead_time_days"], as_of_date)
    z = norm.ppf(min(max(ratio, 1e-6), 1 - 1e-6))
    target_level = max(0.0, mu + z * sigma)

    raw_need = max(0.0, target_level - inventory_position)
    quantity = 0.0 if raw_need <= 0 else max(raw_need, offer["moq"])
    post_order_level = inventory_position + quantity
    profit = _expected_profit(post_order_level, mu, sigma, margin, co, cu)

    return {
        "supplier_id": supplier_id,
        "offer": offer,
        "margin": margin,
        "holding_cost": round(co, 4),
        "stockout_cost": cu,
        "critical_ratio": round(ratio, 4),
        "mu": round(mu, 2),
        "sigma": round(sigma, 2),
        "target_level": round(target_level, 2),
        "quantity": quantity,
        "order_value": round(quantity * offer["unit_price"], 2),
        "expected_profit": round(profit, 2),
        # profit above is "profit from one order cycle", and cycles are NOT the same
        # length across suppliers (10 days of demand for A vs. 3 for B) -- comparing
        # raw expected_profit would structurally favor whichever supplier has the
        # longer lead time, simply because its order covers more total demand, not
        # because it's more profitable. Normalizing by lead_time_days makes it a fair
        # profit-per-day-of-operation comparison, which is what actually decides the
        # supplier below.
        "expected_profit_per_day": round(profit / offer["lead_time_days"], 2),
    }


def _trailing_cost_basis(sku: str, unit_price: float, as_of_date: pd.Timestamp) -> float:
    """Spend-cap basis. Cold start (no 6-month real PO history -- always
    true within this project's 78-day simulated window): estimate from real
    M5 sales history, per CLAUDE.md. Once the real ledger spans the full
    trailing window, use its actual cost instead."""
    window_start = as_of_date - pd.Timedelta(days=TRAILING_COST_WINDOW_DAYS)
    ledger = orders.all_orders(sku)
    ledger_covers_window = not ledger.empty and ledger["order_date"].min() <= window_start
    if ledger_covers_window:
        in_window = ledger[(ledger["order_date"] >= window_start) & (ledger["order_date"] < as_of_date)]
        return float(in_window["total_cost"].sum())

    wide = io.load_sales_wide()
    trailing_units = wide.loc[(wide.index >= window_start) & (wide.index < as_of_date), sku].sum()
    return float(trailing_units) * unit_price


def _is_large(order_value: float, cap: float) -> bool:
    return order_value > LARGE_ORDER_FRACTION * cap


def _other_large_orders_in_window(sku: str, as_of_date: pd.Timestamp, lead_time_days: int, cap: float) -> pd.DataFrame:
    ledger = orders.all_orders(sku)
    if ledger.empty:
        return ledger
    window_start = as_of_date - pd.Timedelta(days=lead_time_days)
    in_window = ledger[
        (ledger["order_date"] >= window_start)
        & (ledger["order_date"] <= as_of_date)
        & (ledger["status"] != "cancelled")
    ]
    return in_window[in_window["total_cost"] > LARGE_ORDER_FRACTION * cap]


def _duplicate_pending(sku: str, supplier_id: str, quantity: float) -> bool:
    pending = orders.pending_orders(sku)
    if pending.empty:
        return False
    same_supplier = pending[pending["supplier_id"] == supplier_id]
    if same_supplier.empty:
        return False
    tolerance = quantity * DUPLICATE_QTY_TOLERANCE
    return bool((same_supplier["quantity"].sub(quantity).abs() <= tolerance).any())


def _ensure_log() -> pd.DataFrame:
    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DECISIONS_PATH.exists():
        pd.DataFrame(columns=DECISION_COLUMNS).to_csv(DECISIONS_PATH, index=False)
    return pd.read_csv(DECISIONS_PATH, dtype={"po_id": str})


def _log_decision(row: dict) -> None:
    log = _ensure_log()
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(DECISIONS_PATH, index=False)


def _already_decided(date: str, sku: str) -> dict | None:
    log = _ensure_log()
    match = log[(log["date"] == date) & (log["sku"] == sku)]
    return match.iloc[-1].to_dict() if not match.empty else None


def run(as_of_date: str | None = None, sku: str | None = None, dry_run: bool = False, force: bool = False) -> dict:
    sku = sku or pos.SKU
    state = inventory.load_state()
    as_of_date = as_of_date or state["current_date"]
    as_of = pd.Timestamp(as_of_date)

    if not force:
        cached = _already_decided(str(as_of.date()), sku)
        if cached is not None:
            return {**cached, "idempotent_replay": True}

    pending = orders.pending_orders(sku)
    pipeline_qty = float(pending["quantity"].sum()) if not pending.empty else 0.0
    inventory_position = state["on_hand"] + pipeline_qty

    candidates = [_evaluate_supplier(sku, sid, as_of, inventory_position) for sid in SUPPLIER_IDS]
    orderable = [c for c in candidates if c["quantity"] > 0]

    base_row = {"date": str(as_of.date()), "sku": sku, "po_id": None}

    if not orderable:
        decision = {
            **base_row,
            "action": "no_order",
            "supplier_id": None,
            "quantity": 0,
            "unit_price": None,
            "order_value": 0.0,
            "expected_profit": None,
            "expected_profit_per_day": None,
            "rationale": (
                f"Inventory position {inventory_position:.1f} already at/above the optimal "
                f"order-up-to level for both suppliers; no order needed."
            ),
        }
        _log_decision(decision)
        return decision

    best = max(orderable, key=lambda c: c["expected_profit_per_day"])
    cap = _trailing_cost_basis(sku, best["offer"]["unit_price"], as_of)

    if best["order_value"] > cap:
        capped_qty = (cap / best["offer"]["unit_price"]) if cap > 0 else 0.0
        if capped_qty < best["offer"]["moq"]:
            decision = {
                **base_row,
                "action": "blocked_spend_cap",
                "supplier_id": best["supplier_id"],
                "quantity": best["quantity"],
                "unit_price": best["offer"]["unit_price"],
                "order_value": best["order_value"],
                "expected_profit": best["expected_profit"],
                "expected_profit_per_day": best["expected_profit_per_day"],
                "rationale": (
                    f"Needed {best['quantity']:.0f} units (${best['order_value']:.2f}) from "
                    f"{best['supplier_id']}, but the spend cap (${cap:.2f}, 50% of trailing-180d "
                    f"cost basis) only allows {capped_qty:.0f} units -- below MOQ "
                    f"{best['offer']['moq']}. Blocked this cycle."
                ),
            }
            _log_decision(decision)
            return decision
        best = {**best, "quantity": capped_qty, "order_value": round(capped_qty * best["offer"]["unit_price"], 2)}

    if _duplicate_pending(sku, best["supplier_id"], best["quantity"]):
        decision = {
            **base_row,
            "action": "duplicate_skipped",
            "supplier_id": best["supplier_id"],
            "quantity": best["quantity"],
            "unit_price": best["offer"]["unit_price"],
            "order_value": best["order_value"],
            "expected_profit": best["expected_profit"],
            "expected_profit_per_day": best["expected_profit_per_day"],
            "rationale": (
                f"A pending order for {sku} from {best['supplier_id']} within "
                f"{DUPLICATE_QTY_TOLERANCE:.0%} of {best['quantity']:.0f} units is already in "
                f"transit; skipped to avoid double-ordering."
            ),
        }
        _log_decision(decision)
        return decision

    is_large = _is_large(best["order_value"], cap)
    if is_large:
        other_large = _other_large_orders_in_window(sku, as_of, best["offer"]["lead_time_days"], cap)
        if not other_large.empty:
            po = orders.place_order(
                sku,
                best["supplier_id"],
                round(best["quantity"]),
                best["offer"]["unit_price"],
                str(as_of.date()),
                best["offer"]["lead_time_days"],
                rationale=(
                    f"Large order (${best['order_value']:.2f} > 70% of ${cap:.2f} cap) with "
                    f"{len(other_large)} other large order(s) already in the "
                    f"{best['offer']['lead_time_days']}-day lead-time window "
                    f"({', '.join(other_large['po_id'])}) -- queued for human approval, not auto-placed."
                ),
                status="pending_approval",
            )
            decision = {
                **base_row,
                "action": "pending_approval",
                "supplier_id": best["supplier_id"],
                "quantity": best["quantity"],
                "unit_price": best["offer"]["unit_price"],
                "order_value": best["order_value"],
                "expected_profit": best["expected_profit"],
                "expected_profit_per_day": best["expected_profit_per_day"],
                "po_id": po.po_id,
                "rationale": po.rationale,
            }
            _log_decision(decision)
            return decision

    other = candidates[1 - SUPPLIER_IDS.index(best["supplier_id"])]
    rationale = (
        f"Chose {best['supplier_id']} ({best['offer']['name']}): order-up-to {best['target_level']:.1f} "
        f"(mu={best['mu']:.1f}, sigma={best['sigma']:.1f}, critical ratio={best['critical_ratio']:.3f}) "
        f"-> order {best['quantity']:.0f} units @ ${best['offer']['unit_price']}, expected profit "
        f"${best['expected_profit']:.2f} (${best['expected_profit_per_day']:.2f}/day over its "
        f"{best['offer']['lead_time_days']}-day lead time) vs. {other['supplier_id']} at "
        f"${other['expected_profit']:.2f} (${other['expected_profit_per_day']:.2f}/day over "
        f"{other['offer']['lead_time_days']} days) -- compared per-day since the two suppliers' "
        f"order cycles cover different amounts of demand."
    )

    po_id = None
    if not dry_run:
        po = orders.place_order(
            sku,
            best["supplier_id"],
            round(best["quantity"]),
            best["offer"]["unit_price"],
            str(as_of.date()),
            best["offer"]["lead_time_days"],
            rationale=rationale,
        )
        po_id = po.po_id

    decision = {
        **base_row,
        "action": "dry_run_order" if dry_run else "order",
        "supplier_id": best["supplier_id"],
        "quantity": best["quantity"],
        "unit_price": best["offer"]["unit_price"],
        "order_value": best["order_value"],
        "expected_profit": best["expected_profit"],
        "expected_profit_per_day": best["expected_profit_per_day"],
        "po_id": po_id,
        "rationale": rationale,
    }
    _log_decision(decision)
    return decision


def load_decisions() -> pd.DataFrame:
    return _ensure_log()


if __name__ == "__main__":
    result = run(dry_run="--dry-run" in sys.argv)
    print(f"[{result['date']}] {result['action']}: {result['rationale']}")
