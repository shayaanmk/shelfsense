"""Two hardcoded supplier offers for commerce.pos.SKU (build plan step 7).

Per CLAUDE.md's already-decided methodology: derive both as a discount off
the SKU's real average sell price (commerce.pos.average_sell_price(), $4.98
for FOODS_3_120) rather than picking arbitrary dollar figures, so margins
stay realistic. Supplier A = cheap + slow, Supplier B = pricier + fast.

Discount/MOQ/lead-time numbers (own these, per CLAUDE.md -- flag for
step 8's sign-off, not silently final):
  - Supplier A: 30% off retail ($3.49/unit), 10-day lead time, MOQ 150 --
    a bulk/overseas-style supplier.
  - Supplier B: 20% off retail ($3.98/unit), 3-day lead time, MOQ 75 -- a
    regional/premium supplier paying more for speed.
  Both land in a "normal grocery margin" range (20-30% off retail), unlike
  picking numbers that would make one supplier obviously dominant.

Flag for step 8 (computed here, decided there): margin per unit is ~$1.00-
1.49, versus CLAUDE.md's $45/unit stockout-cost baseline -- 30-45x the
margin, pushing the newsvendor critical ratio Cu/(Cu+Co) to ~0.99 regardless
of which supplier's lead time is used (see main() below for the actual
numbers). CLAUDE.md already flagged $45 as "worth sanity-checking against
the actual chosen SKU rather than treating $15/$45 as universal" -- this is
that check, with FOODS_3_120's real numbers, worse than CLAUDE.md's own
illustrative example. Revisit $45 (or accept the always-over-order bias as
intentional) when building the restocking agent's objective in step 8.
"""

from __future__ import annotations

from commerce.pos import average_sell_price

DISCOUNT_A = 0.30
DISCOUNT_B = 0.20

HOLDING_COST_PER_UNIT_PER_YEAR = 15.0
STOCKOUT_COST_PER_UNIT = 45.0


def _offers() -> dict:
    retail = average_sell_price()
    return {
        "supplier_a": {
            "supplier_id": "supplier_a",
            "name": "Supplier A (bulk, slow)",
            "unit_price": round(retail * (1 - DISCOUNT_A), 2),
            "moq": 150,
            "lead_time_days": 10,
        },
        "supplier_b": {
            "supplier_id": "supplier_b",
            "name": "Supplier B (premium, fast)",
            "unit_price": round(retail * (1 - DISCOUNT_B), 2),
            "moq": 75,
            "lead_time_days": 3,
        },
    }


def list_offers() -> list[dict]:
    return list(_offers().values())


def get_offer(supplier_id: str) -> dict:
    offers = _offers()
    if supplier_id not in offers:
        raise ValueError(f"Unknown supplier {supplier_id!r}; choices are {list(offers)}.")
    return offers[supplier_id]


def margin_per_unit(supplier_id: str) -> float:
    return round(average_sell_price() - get_offer(supplier_id)["unit_price"], 2)


def critical_ratio(lead_time_days: int, stockout_cost: float = STOCKOUT_COST_PER_UNIT) -> float:
    """Cu / (Cu + Co), Co prorated to the order cycle per CLAUDE.md's
    already-decided formula: Co = holding_cost_per_year * (cycle_days/365)."""
    holding_cost = HOLDING_COST_PER_UNIT_PER_YEAR * (lead_time_days / 365)
    return stockout_cost / (stockout_cost + holding_cost)


if __name__ == "__main__":
    retail = average_sell_price()
    print(f"Retail price: ${retail}\n")
    for offer in list_offers():
        margin = margin_per_unit(offer["supplier_id"])
        ratio = critical_ratio(offer["lead_time_days"])
        print(
            f"{offer['name']}: ${offer['unit_price']}/unit, MOQ {offer['moq']}, "
            f"lead time {offer['lead_time_days']}d -- margin ${margin}/unit, "
            f"critical ratio {ratio:.3f}"
        )
