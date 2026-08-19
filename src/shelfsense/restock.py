from __future__ import annotations

from dataclasses import dataclass

from shelfsense.inventory import Inventory


@dataclass(frozen=True)
class RestockSuggestion:
    sku: str
    units: int
    cost: float


@dataclass(frozen=True)
class RestockPlan:
    suggestions: tuple[RestockSuggestion, ...]

    @property
    def total_units(self) -> int:
        return sum(s.units for s in self.suggestions)

    @property
    def total_cost(self) -> float:
        return sum(s.cost for s in self.suggestions)

    def __bool__(self) -> bool:
        return bool(self.suggestions)


def plan_restock(
    inventory: Inventory,
    threshold: float = 0.25,
    target_fill: float = 1.0,
    budget: float | None = None,
) -> RestockPlan:
    """Suggest refills for low-stock slots, emptiest first, within an optional budget."""
    if not 0 < target_fill <= 1:
        raise ValueError("target_fill must be in (0, 1]")
    if budget is not None and budget < 0:
        raise ValueError("budget must be non-negative")

    remaining = budget
    suggestions: list[RestockSuggestion] = []
    for slot in inventory.low_stock(threshold):
        target_quantity = int(slot.capacity * target_fill)
        units = max(target_quantity - slot.quantity, 0)
        if units == 0:
            continue
        price = slot.product.unit_price
        if remaining is not None:
            affordable = slot.capacity if price == 0 else int(remaining // price)
            units = min(units, affordable)
            if units == 0:
                continue
            remaining -= units * price
        suggestions.append(RestockSuggestion(slot.sku, units, units * price))
    return RestockPlan(tuple(suggestions))
