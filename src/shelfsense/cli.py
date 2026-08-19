from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from shelfsense.inventory import Inventory
from shelfsense.models import Product, ShelfSlot
from shelfsense.restock import plan_restock


def load_inventory(path: str) -> Inventory:
    """Load shelf slots from a JSON file of {sku, name, unit_price, quantity, capacity}."""
    with open(path) as handle:
        raw = json.load(handle)
    slots = [
        ShelfSlot(
            product=Product(entry["sku"], entry["name"], float(entry["unit_price"])),
            quantity=int(entry["quantity"]),
            capacity=int(entry["capacity"]),
        )
        for entry in raw
    ]
    return Inventory(slots)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shelfsense")
    parser.add_argument("path", help="path to inventory JSON file")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--budget", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inventory = load_inventory(args.path)
    plan = plan_restock(inventory, threshold=args.threshold, budget=args.budget)
    if not plan:
        print("No restock needed")
        return 0
    for suggestion in plan.suggestions:
        print(f"{suggestion.sku}: {suggestion.units} units (${suggestion.cost:.2f})")
    print(f"total: {plan.total_units} units (${plan.total_cost:.2f})")
    return 0
