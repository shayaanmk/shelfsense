import pytest

from shelfsense.inventory import Inventory
from shelfsense.models import Product, ShelfSlot
from shelfsense.restock import RestockPlan, plan_restock


def make_slot(sku: str, quantity: int, capacity: int, price: float) -> ShelfSlot:
    return ShelfSlot(Product(sku, f"Item {sku}", price), quantity, capacity)


@pytest.fixture
def inventory() -> Inventory:
    return Inventory(
        [
            make_slot("A", 1, 10, 2.0),
            make_slot("B", 9, 10, 3.0),
            make_slot("C", 0, 4, 5.0),
        ]
    )


def test_plan_refills_low_stock_to_capacity(inventory):
    plan = plan_restock(inventory)
    assert [(s.sku, s.units, s.cost) for s in plan.suggestions] == [
        ("C", 4, 20.0),
        ("A", 9, 18.0),
    ]
    assert plan.total_units == 13
    assert plan.total_cost == pytest.approx(38.0)
    assert bool(plan) is True


def test_empty_plan_is_falsy():
    plan = plan_restock(Inventory([make_slot("A", 10, 10, 1.0)]))
    assert plan == RestockPlan(())
    assert not plan
    assert plan.total_units == 0
    assert plan.total_cost == 0


def test_target_fill_partial(inventory):
    plan = plan_restock(inventory, target_fill=0.5)
    assert [(s.sku, s.units) for s in plan.suggestions] == [("C", 2), ("A", 4)]


def test_slot_already_above_target_is_skipped(inventory):
    plan = plan_restock(inventory, threshold=1.0, target_fill=0.5)
    assert [(s.sku, s.units) for s in plan.suggestions] == [("C", 2), ("A", 4)]


def test_budget_truncates_and_skips(inventory):
    plan = plan_restock(inventory, budget=25.0)
    assert [(s.sku, s.units, s.cost) for s in plan.suggestions] == [
        ("C", 4, 20.0),
        ("A", 2, 4.0),
    ]
    assert plan.total_cost <= 25.0


def test_zero_budget_yields_empty_plan(inventory):
    assert not plan_restock(inventory, budget=0.0)


def test_free_items_fill_regardless_of_budget():
    inventory = Inventory([make_slot("F", 0, 6, 0.0)])
    plan = plan_restock(inventory, budget=0.0)
    assert [(s.sku, s.units, s.cost) for s in plan.suggestions] == [("F", 6, 0.0)]


@pytest.mark.parametrize("target_fill", [0.0, -0.5, 1.5])
def test_target_fill_validated(inventory, target_fill):
    with pytest.raises(ValueError):
        plan_restock(inventory, target_fill=target_fill)


def test_negative_budget_rejected(inventory):
    with pytest.raises(ValueError):
        plan_restock(inventory, budget=-1.0)
