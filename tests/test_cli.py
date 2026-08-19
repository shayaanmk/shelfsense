import json

import pytest

from shelfsense.cli import build_parser, load_inventory, main

INVENTORY = [
    {"sku": "A", "name": "Item A", "unit_price": 2.0, "quantity": 1, "capacity": 10},
    {"sku": "B", "name": "Item B", "unit_price": 3.0, "quantity": 10, "capacity": 10},
]


@pytest.fixture
def inventory_file(tmp_path):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(INVENTORY))
    return str(path)


def test_load_inventory(inventory_file):
    inventory = load_inventory(inventory_file)
    assert len(inventory) == 2
    assert inventory.quantity("A") == 1
    assert inventory.slot("B").product.name == "Item B"


def test_parser_defaults(inventory_file):
    args = build_parser().parse_args([inventory_file])
    assert args.threshold == 0.25
    assert args.budget is None


def test_main_prints_suggestions(inventory_file, capsys):
    assert main([inventory_file]) == 0
    out = capsys.readouterr().out
    assert "A: 9 units ($18.00)" in out
    assert "total: 9 units ($18.00)" in out


def test_main_reports_no_restock_needed(inventory_file, capsys):
    assert main([inventory_file, "--threshold", "0.0"]) == 0
    assert capsys.readouterr().out.strip() == "No restock needed"


def test_main_honours_budget(inventory_file, capsys):
    assert main([inventory_file, "--budget", "4"]) == 0
    assert "A: 2 units ($4.00)" in capsys.readouterr().out


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inventory(str(tmp_path / "nope.json"))
