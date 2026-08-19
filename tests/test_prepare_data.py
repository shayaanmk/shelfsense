"""Tests for prepare_data's raw-data validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting import prepare_data

N_DAYS = 40
ITEMS = [f"FOODS_1_{i:03d}" for i in range(3)]


def _write_dataset(raw: Path, sales_edit=None, calendar_edit=None) -> None:
    rng = np.random.default_rng(0)
    sales = pd.DataFrame(
        [
            {
                "id": f"{item}_CA_1_evaluation",
                "item_id": item,
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": "CA_1",
                "state_id": "CA",
                **{f"d_{d}": int(rng.integers(0, 20)) for d in range(1, N_DAYS + 1)},
            }
            for item in ITEMS
        ]
    )
    calendar = pd.DataFrame(
        {
            "d": [f"d_{d}" for d in range(1, N_DAYS + 1)],
            "date": pd.date_range("2015-01-01", periods=N_DAYS).astype(str),
            "wm_yr_wk": 11101,
            "wday": 1,
            "month": 1,
            "year": 2015,
            "event_name_1": None,
            "event_type_1": None,
            "snap_CA": 0,
        }
    )
    if sales_edit is not None:
        sales = sales_edit(sales)
    if calendar_edit is not None:
        calendar = calendar_edit(calendar)

    raw.mkdir(parents=True, exist_ok=True)
    sales.to_csv(raw / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(raw / "calendar.csv", index=False)
    pd.DataFrame(
        {"store_id": ["CA_1"], "item_id": [ITEMS[0]], "wm_yr_wk": [11101], "sell_price": [1.0]}
    ).to_csv(raw / "sell_prices.csv", index=False)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data, "RAW", tmp_path / "raw")
    monkeypatch.setattr(prepare_data, "PROCESSED", tmp_path / "processed")
    monkeypatch.setitem(prepare_data.CONFIG, "n_skus", len(ITEMS))
    return lambda **edits: _write_dataset(tmp_path / "raw", **edits)


def test_happy_path_writes_numeric_units(dataset, tmp_path):
    dataset()
    prepare_data.main()
    out = pd.read_parquet(tmp_path / "processed" / "sales_long.parquet")
    assert len(out) == len(ITEMS) * N_DAYS
    assert out["units"].dtype.kind == "i"
    assert out["date"].notna().all()


def test_missing_raw_files(dataset, tmp_path):
    (tmp_path / "raw").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Download the M5 dataset"):
        prepare_data.main()


def test_non_numeric_units_are_rejected(dataset):
    def corrupt(sales):
        sales["d_5"] = sales["d_5"].astype(object)
        sales.loc[0, "d_5"] = "abc"
        return sales

    dataset(sales_edit=corrupt)
    with pytest.raises(ValueError, match="non-numeric unit value"):
        prepare_data.main()


def test_unmatched_day_column_is_rejected(dataset):
    dataset(calendar_edit=lambda cal: cal[cal["d"] != "d_5"])
    with pytest.raises(ValueError, match=r"no row in calendar\.csv"):
        prepare_data.main()


def test_missing_snap_column_is_rejected(dataset):
    dataset(calendar_edit=lambda cal: cal.drop(columns=["snap_CA"]))
    with pytest.raises(ValueError, match=r"missing expected column\(s\) \['snap_CA'\]"):
        prepare_data.main()


def test_unknown_store_lists_available_values(dataset, monkeypatch):
    dataset()
    monkeypatch.setitem(prepare_data.CONFIG, "store", "ZZ_9")
    with pytest.raises(ValueError, match="Available categories"):
        prepare_data.main()
