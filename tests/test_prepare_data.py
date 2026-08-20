"""Tests for prepare_data's raw-data validation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting import prepare_data

REPO_ROOT = Path(__file__).resolve().parent.parent

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


# --- Sales-file discovery and the reshaping done by main() --------------------

DAY_COLS = ["d_1", "d_2", "d_3"]


def _small_sales() -> pd.DataFrame:
    """Two FOODS/CA_1 SKUs with different volumes, plus rows that must be filtered out."""
    rows = [
        ("FOODS_1_001_CA_1_evaluation", "FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA", 5, 5, 5),
        ("FOODS_1_002_CA_1_evaluation", "FOODS_1_002", "FOODS_1", "FOODS", "CA_1", "CA", 1, 0, 2),
        ("FOODS_1_003_CA_1_evaluation", "FOODS_1_003", "FOODS_1", "FOODS", "CA_1", "CA", 0, 0, 1),
        ("FOODS_1_004_CA_2_evaluation", "FOODS_1_004", "FOODS_1", "FOODS", "CA_2", "CA", 9, 9, 9),
        ("HOBBIES_1_1_CA_1_evaluation", "HOBBIES_1_1", "HOBBIES_1", "HOBBIES", "CA_1", "CA", 7, 7, 7),
    ]
    return pd.DataFrame(rows, columns=[*prepare_data.SALES_ID_COLS, *DAY_COLS])


def _small_calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "d": DAY_COLS,
            "date": ["2011-01-29", "2011-01-30", "2011-01-31"],
            "wm_yr_wk": [11101, 11101, 11101],
            "wday": [1, 2, 3],
            "month": [1, 1, 1],
            "year": [2011, 2011, 2011],
            "event_name_1": [None, "SuperBowl", None],
            "event_type_1": [None, "Sporting", None],
            "snap_CA": [0, 1, 1],
            "snap_TX": [0, 0, 1],
            "snap_WI": [1, 1, 0],
        }
    )


@pytest.fixture
def small_dataset(tmp_path, monkeypatch):
    """Repoint the module's paths at a 3-day synthetic slice and write it out."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    monkeypatch.setattr(prepare_data, "RAW", raw)
    monkeypatch.setattr(prepare_data, "PROCESSED", processed)

    _small_sales().to_csv(raw / "sales_train_evaluation.csv", index=False)
    _small_calendar().to_csv(raw / "calendar.csv", index=False)
    pd.DataFrame({"store_id": ["CA_1"], "item_id": ["FOODS_1_001"]}).to_csv(
        raw / "sell_prices.csv", index=False
    )
    return raw, processed


class TestSalesFileSelection:
    """Which of the two M5 sales releases main() actually reads."""

    def test_prefers_evaluation_over_validation(self, small_dataset, monkeypatch):
        raw, processed = small_dataset
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 1)
        # an older release naming its SKUs differently; the evaluation file must win
        _small_sales().assign(item_id=lambda df: df["item_id"] + "_OLD").to_csv(
            raw / "sales_train_validation.csv", index=False
        )

        prepare_data.main()

        assert (processed / "skus.txt").read_text(encoding="utf-8").splitlines() == ["FOODS_1_001"]

    def test_falls_back_to_validation(self, small_dataset, monkeypatch):
        raw, processed = small_dataset
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 1)
        (raw / "sales_train_evaluation.csv").rename(raw / "sales_train_validation.csv")

        prepare_data.main()

        assert (processed / "skus.txt").read_text(encoding="utf-8").splitlines() == ["FOODS_1_001"]

    def test_missing_sales_file_raises(self, small_dataset):
        raw, _ = small_dataset
        (raw / "sales_train_evaluation.csv").unlink()
        with pytest.raises(FileNotFoundError, match="Download the M5 dataset"):
            prepare_data.main()


class TestReshaping:
    def test_writes_long_parquet_and_sku_list(self, small_dataset, monkeypatch, capsys):
        _, processed = small_dataset
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 2)

        prepare_data.main()

        long = pd.read_parquet(processed / "sales_long.parquet")
        # top 2 SKUs of the FOODS/CA_1 slice, one row per (item, day)
        assert sorted(long["item_id"].unique()) == ["FOODS_1_001", "FOODS_1_002"]
        assert len(long) == 2 * len(DAY_COLS)
        assert long["date"].dtype.kind == "M"
        assert long["date"].min() == pd.Timestamp("2011-01-29")
        assert long[["item_id", "date"]].equals(
            long.sort_values(["item_id", "date"]).reset_index(drop=True)[["item_id", "date"]]
        )
        skus = (processed / "skus.txt").read_text(encoding="utf-8").splitlines()
        assert skus == ["FOODS_1_001", "FOODS_1_002"]
        assert "Kept 2 SKUs" in capsys.readouterr().out

    def test_snap_column_is_selected_by_store_and_renamed(self, small_dataset, monkeypatch):
        _, processed = small_dataset
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 1)

        prepare_data.main()

        long = pd.read_parquet(processed / "sales_long.parquet").sort_values("date")
        assert "snap" in long.columns
        assert not [c for c in long.columns if c.startswith("snap_")]
        assert long["snap"].tolist() == [0, 1, 1]
        assert long["event_name_1"].tolist()[1] == "SuperBowl"

    def test_units_survive_the_wide_to_long_melt(self, small_dataset, monkeypatch):
        _, processed = small_dataset
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 2)

        prepare_data.main()

        by_item = pd.read_parquet(processed / "sales_long.parquet").groupby("item_id")["units"].sum()
        assert by_item["FOODS_1_001"] == 15
        assert by_item["FOODS_1_002"] == 3

    def test_scoping_to_another_store(self, small_dataset, monkeypatch, capsys):
        _, processed = small_dataset
        monkeypatch.setitem(prepare_data.CONFIG, "store", "CA_2")
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 5)

        prepare_data.main()

        long = pd.read_parquet(processed / "sales_long.parquet")
        assert long["item_id"].unique().tolist() == ["FOODS_1_004"]
        # fewer SKUs than requested is a warning, not a failure
        assert "only 1 exist" in capsys.readouterr().err

    def test_empty_slice_raises(self, small_dataset, monkeypatch):
        monkeypatch.setitem(prepare_data.CONFIG, "store", "WI_3")
        with pytest.raises(ValueError, match="No rows for category=FOODS store=WI_3"):
            prepare_data.main()

    @pytest.mark.parametrize("missing", ["calendar.csv", "sell_prices.csv"])
    def test_missing_companion_csv_raises(self, small_dataset, missing):
        raw, _ = small_dataset
        (raw / missing).unlink()
        with pytest.raises(FileNotFoundError, match="unzip the CSVs into data/raw/"):
            prepare_data.main()

    def test_creates_processed_directory(self, small_dataset, monkeypatch):
        _, processed = small_dataset
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 1)
        assert not processed.exists()

        prepare_data.main()

        assert processed.is_dir()


class TestGuardRailMessages:
    def test_sales_file_without_day_columns_is_rejected(self, small_dataset):
        raw, _ = small_dataset
        _small_sales().drop(columns=DAY_COLS).to_csv(
            raw / "sales_train_evaluation.csv", index=False
        )
        with pytest.raises(ValueError, match=r"no d_\* day columns"):
            prepare_data.main()


def test_module_entrypoint_exits_nonzero_with_a_readable_error(tmp_path):
    """`python -m forecasting.prepare_data` maps the guard rails onto exit code 1."""
    proc = subprocess.run(
        [sys.executable, "-m", "forecasting.prepare_data"],
        cwd=tmp_path,  # no data/raw here
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 1
    assert proc.stderr.startswith("ERROR: ")
    assert "Download the M5 dataset" in proc.stderr
