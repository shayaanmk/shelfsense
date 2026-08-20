"""Tests for the shared dataset locations and loaders in forecasting.io.

Every step of the pipeline goes through this module for its paths, its
"did you unzip the M5 CSVs?" error and its processed-data validation, so the
guard rails are asserted here once instead of per-step.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forecasting import io


def _long_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-01-02", "2015-01-01", "2015-01-02", "2015-01-01"]),
            "item_id": ["A", "A", "B", "B"],
            "units": [2.0, 1.0, 20.0, 10.0],
        }
    )


def _write_long(processed, frame: pd.DataFrame | None = None) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    (frame if frame is not None else _long_frame()).to_parquet(
        processed / io.SALES_LONG_NAME, index=False
    )


class TestRawFileLookup:
    def test_require_raw_file_returns_the_path(self, tmp_path):
        (tmp_path / io.CALENDAR).touch()
        assert io.require_raw_file(io.CALENDAR, tmp_path) == tmp_path / io.CALENDAR

    def test_require_raw_file_names_the_file_and_the_download_hint(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            io.require_raw_file(io.PRICES, tmp_path)
        assert str(tmp_path / io.PRICES) in str(exc.value)
        assert "unzip the CSVs into data/raw/" in str(exc.value)

    def test_require_any_raw_file_prefers_the_earlier_candidate(self, tmp_path):
        for name in io.SALES_CANDIDATES:
            (tmp_path / name).touch()
        assert io.require_any_raw_file(io.SALES_CANDIDATES, tmp_path).name == io.SALES_CANDIDATES[0]

    def test_require_any_raw_file_falls_back_to_a_later_candidate(self, tmp_path):
        (tmp_path / io.SALES_CANDIDATES[1]).touch()
        assert io.require_any_raw_file(io.SALES_CANDIDATES, tmp_path).name == io.SALES_CANDIDATES[1]

    def test_require_any_raw_file_lists_every_candidate_when_none_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            io.require_any_raw_file(io.SALES_CANDIDATES, tmp_path)
        for name in io.SALES_CANDIDATES:
            assert name in str(exc.value)
        assert "Download the M5 dataset" in str(exc.value)

    def test_raw_dir_defaults_to_the_module_constant(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io, "RAW", tmp_path)
        (tmp_path / io.CALENDAR).touch()
        assert io.require_raw_file(io.CALENDAR) == tmp_path / io.CALENDAR
        assert io.require_any_raw_file((io.CALENDAR,)) == tmp_path / io.CALENDAR


class TestProcessedPaths:
    def test_ensure_processed_creates_missing_parents_and_returns_the_dir(self, tmp_path):
        target = tmp_path / "nested" / "processed"
        assert io.ensure_processed(target) == target
        assert target.is_dir()

    def test_ensure_processed_is_idempotent(self, tmp_path):
        io.ensure_processed(tmp_path)
        assert io.ensure_processed(tmp_path) == tmp_path

    def test_ensure_processed_defaults_to_the_module_constant(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io, "PROCESSED", tmp_path / "processed")
        assert io.ensure_processed() == tmp_path / "processed"
        assert (tmp_path / "processed").is_dir()

    @pytest.mark.parametrize(
        ("helper", "name"),
        [
            (io.sales_long_path, io.SALES_LONG_NAME),
            (io.skus_path, io.SKUS_NAME),
            (io.baseline_metrics_path, io.BASELINE_METRICS_NAME),
        ],
    )
    def test_artifact_paths_live_in_the_given_dir(self, tmp_path, helper, name):
        assert helper(tmp_path) == tmp_path / name

    @pytest.mark.parametrize(
        ("helper", "name"),
        [
            (io.sales_long_path, io.SALES_LONG_NAME),
            (io.skus_path, io.SKUS_NAME),
            (io.baseline_metrics_path, io.BASELINE_METRICS_NAME),
        ],
    )
    def test_artifact_paths_default_to_the_module_constant(self, tmp_path, monkeypatch, helper, name):
        monkeypatch.setattr(io, "PROCESSED", tmp_path)
        assert helper() == tmp_path / name


class TestLoadSalesLong:
    def test_reads_the_parquet_written_by_prepare_data(self, tmp_path):
        _write_long(tmp_path)
        long_df = io.load_sales_long(tmp_path)
        assert len(long_df) == 4
        assert set(long_df.columns) == {"date", "item_id", "units"}

    def test_missing_parquet_points_at_prepare_data(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run `python -m forecasting.prepare_data`"):
            io.load_sales_long(tmp_path)

    def test_missing_columns_are_listed(self, tmp_path):
        _write_long(tmp_path, pd.DataFrame({"date": [1], "units": [2.0]}))
        with pytest.raises(ValueError, match=r"missing column\(s\) \['item_id'\]"):
            io.load_sales_long(tmp_path)

    def test_the_values_column_is_validated_too(self, tmp_path):
        _write_long(tmp_path)
        with pytest.raises(ValueError, match=r"missing column\(s\) \['revenue'\]"):
            io.load_sales_long(tmp_path, values="revenue")

    def test_processed_dir_defaults_to_the_module_constant(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io, "PROCESSED", tmp_path)
        _write_long(tmp_path)
        assert len(io.load_sales_long()) == 4


class TestLoadSalesWide:
    def test_pivots_to_a_date_by_item_matrix_sorted_by_date(self, tmp_path):
        _write_long(tmp_path)
        wide = io.load_sales_wide(processed=tmp_path)
        assert list(wide.columns) == ["A", "B"]
        assert list(wide.index) == [pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-02")]
        assert wide["B"].tolist() == [10.0, 20.0]

    def test_duplicate_date_item_pairs_name_the_offending_file(self, tmp_path):
        _write_long(
            tmp_path,
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2015-01-01", "2015-01-01"]),
                    "item_id": ["A", "A"],
                    "units": [1.0, 2.0],
                }
            ),
        )
        with pytest.raises(ValueError, match="duplicate") as exc:
            io.load_sales_wide(processed=tmp_path)
        assert io.SALES_LONG_NAME in str(exc.value)

    def test_reshapes_an_alternative_values_column(self, tmp_path):
        frame = _long_frame().assign(revenue=lambda df: df["units"] * 3)
        _write_long(tmp_path, frame)
        wide = io.load_sales_wide(values="revenue", processed=tmp_path)
        assert wide["A"].tolist() == [3.0, 6.0]

    def test_processed_dir_defaults_to_the_module_constant(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io, "PROCESSED", tmp_path)
        _write_long(tmp_path)
        assert io.load_sales_wide().shape == (2, 2)
