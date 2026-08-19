from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    """A sellable item identified by its SKU."""

    sku: str
    name: str
    unit_price: float

    def __post_init__(self) -> None:
        if not self.sku.strip():
            raise ValueError("sku must be a non-empty string")
        if not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.unit_price < 0:
            raise ValueError("unit_price must be non-negative")


@dataclass
class ShelfSlot:
    """Physical shelf space allocated to a single product."""

    product: Product
    quantity: int
    capacity: int

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if self.quantity > self.capacity:
            raise ValueError("quantity cannot exceed capacity")

    @property
    def sku(self) -> str:
        return self.product.sku

    @property
    def free_space(self) -> int:
        return self.capacity - self.quantity

    @property
    def fill_ratio(self) -> float:
        return self.quantity / self.capacity

    @property
    def is_empty(self) -> bool:
        return self.quantity == 0

    def add(self, amount: int) -> int:
        """Add units up to capacity and return the number actually added."""
        if amount < 0:
            raise ValueError("amount must be non-negative")
        added = min(amount, self.free_space)
        self.quantity += added
        return added

    def remove(self, amount: int) -> int:
        """Remove up to `amount` units and return the number actually removed."""
        if amount < 0:
            raise ValueError("amount must be non-negative")
        removed = min(amount, self.quantity)
        self.quantity -= removed
        return removed
