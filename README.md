# shelfsense

Shelf inventory tracking and restock planning.

## Modules

- `shelfsense.models` — `Product` and `ShelfSlot` (capacity-bounded shelf space).
- `shelfsense.inventory` — `Inventory`, a SKU-keyed collection of slots with sales, restocks, valuation, and low-stock detection.
- `shelfsense.restock` — `plan_restock`, which suggests refills for low-stock slots (emptiest first) within an optional budget.
- `shelfsense.cli` — `shelfsense <inventory.json>` prints a restock plan.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/pytest
```

`pytest` reports branch coverage for `shelfsense` by default.

## Inventory file format

```json
[
  {"sku": "A", "name": "Oat Milk", "unit_price": 2.5, "quantity": 1, "capacity": 10}
]
```
