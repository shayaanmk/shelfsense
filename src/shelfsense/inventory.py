from __future__ import annotations

from collections.abc import Iterator

from shelfsense.models import Product, ShelfSlot


class UnknownSkuError(KeyError):
    """Raised when an operation references a SKU that is not on any shelf."""


class Inventory:
    """Tracks shelf slots keyed by SKU."""

    def __init__(self, slots: list[ShelfSlot] | None = None) -> None:
        self._slots: dict[str, ShelfSlot] = {}
        for slot in slots or []:
            self.add_slot(slot)

    def __len__(self) -> int:
        return len(self._slots)

    def __contains__(self, sku: object) -> bool:
        return sku in self._slots

    def __iter__(self) -> Iterator[ShelfSlot]:
        return iter(self._slots.values())

    def add_slot(self, slot: ShelfSlot) -> None:
        if slot.sku in self._slots:
            raise ValueError(f"slot for sku {slot.sku!r} already exists")
        self._slots[slot.sku] = slot

    def slot(self, sku: str) -> ShelfSlot:
        try:
            return self._slots[sku]
        except KeyError:
            raise UnknownSkuError(sku) from None

    def quantity(self, sku: str) -> int:
        return self.slot(sku).quantity

    def restock(self, sku: str, amount: int) -> int:
        return self.slot(sku).add(amount)

    def record_sale(self, sku: str, amount: int) -> int:
        return self.slot(sku).remove(amount)

    def total_value(self) -> float:
        return sum(slot.quantity * slot.product.unit_price for slot in self._slots.values())

    def low_stock(self, threshold: float = 0.25) -> list[ShelfSlot]:
        """Slots whose fill ratio is at or below `threshold`, emptiest first."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        low = [slot for slot in self._slots.values() if slot.fill_ratio <= threshold]
        return sorted(low, key=lambda slot: (slot.fill_ratio, slot.sku))

    def products(self) -> list[Product]:
        return [slot.product for slot in self._slots.values()]
