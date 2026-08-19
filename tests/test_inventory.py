import pytest

from shelfsense.inventory import Inventory, UnknownSkuError
from shelfsense.models import Product, ShelfSlot


def make_slot(sku: str, quantity: int, capacity: int, price: float = 1.0) -> ShelfSlot:
    return ShelfSlot(Product(sku, f"Item {sku}", price), quantity, capacity)


@pytest.fixture
def inventory() -> Inventory:
    return Inventory(
        [
            make_slot("A", 1, 10, 2.0),
            make_slot("B", 8, 10, 3.0),
            make_slot("C", 0, 4, 5.0),
        ]
    )


def test_empty_inventory():
    inventory = Inventory()
    assert len(inventory) == 0
    assert inventory.total_value() == 0
    assert inventory.low_stock() == []


def test_container_protocol(inventory):
    assert len(inventory) == 3
    assert "A" in inventory
    assert "Z" not in inventory
    assert {slot.sku for slot in inventory} == {"A", "B", "C"}


def test_add_duplicate_slot_rejected(inventory):
    with pytest.raises(ValueError, match="already exists"):
        inventory.add_slot(make_slot("A", 0, 5))


def test_unknown_sku_raises(inventory):
    with pytest.raises(UnknownSkuError):
        inventory.slot("Z")
    with pytest.raises(UnknownSkuError):
        inventory.quantity("Z")


def test_restock_and_sale_return_applied_amounts(inventory):
    assert inventory.restock("A", 20) == 9
    assert inventory.quantity("A") == 10
    assert inventory.record_sale("A", 4) == 4
    assert inventory.record_sale("C", 1) == 0


def test_total_value(inventory):
    assert inventory.total_value() == pytest.approx(1 * 2.0 + 8 * 3.0)


def test_low_stock_sorted_emptiest_first(inventory):
    assert [slot.sku for slot in inventory.low_stock()] == ["C", "A"]
    assert [slot.sku for slot in inventory.low_stock(threshold=1.0)] == ["C", "A", "B"]
    assert inventory.low_stock(threshold=0.0) == [inventory.slot("C")]


def test_low_stock_ties_broken_by_sku():
    inventory = Inventory([make_slot("B", 1, 10), make_slot("A", 1, 10)])
    assert [slot.sku for slot in inventory.low_stock()] == ["A", "B"]


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_low_stock_threshold_validated(inventory, threshold):
    with pytest.raises(ValueError):
        inventory.low_stock(threshold)


def test_products(inventory):
    assert [product.sku for product in inventory.products()] == ["A", "B", "C"]
