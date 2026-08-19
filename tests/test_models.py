import pytest

from shelfsense.models import Product, ShelfSlot


@pytest.mark.parametrize(
    ("sku", "name", "price"),
    [
        ("", "Oat Milk", 1.0),
        ("  ", "Oat Milk", 1.0),
        ("SKU1", " ", 1.0),
        ("SKU1", "Oat Milk", -0.01),
    ],
)
def test_product_rejects_invalid_fields(sku, name, price):
    with pytest.raises(ValueError):
        Product(sku, name, price)


def test_product_allows_zero_price():
    assert Product("SKU1", "Sample", 0.0).unit_price == 0.0


@pytest.mark.parametrize(("quantity", "capacity"), [(0, 0), (0, -1), (-1, 5), (6, 5)])
def test_shelf_slot_rejects_invalid_fields(product, quantity, capacity):
    with pytest.raises(ValueError):
        ShelfSlot(product=product, quantity=quantity, capacity=capacity)


def test_shelf_slot_derived_properties(slot):
    assert slot.sku == "SKU1"
    assert slot.free_space == 6
    assert slot.fill_ratio == pytest.approx(0.4)
    assert slot.is_empty is False


def test_is_empty_when_quantity_zero(product):
    assert ShelfSlot(product=product, quantity=0, capacity=3).is_empty is True


def test_add_caps_at_capacity(slot):
    assert slot.add(3) == 3
    assert slot.quantity == 7
    assert slot.add(99) == 3
    assert slot.quantity == slot.capacity


def test_add_zero_is_noop(slot):
    assert slot.add(0) == 0
    assert slot.quantity == 4


def test_remove_caps_at_available_quantity(slot):
    assert slot.remove(3) == 3
    assert slot.quantity == 1
    assert slot.remove(99) == 1
    assert slot.is_empty


@pytest.mark.parametrize("method", ["add", "remove"])
def test_negative_amounts_rejected(slot, method):
    with pytest.raises(ValueError):
        getattr(slot, method)(-1)
