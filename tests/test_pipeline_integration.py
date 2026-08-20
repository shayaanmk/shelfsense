"""End-to-end tests of the pipeline: raw M5 CSVs -> parquet -> baseline metrics.

The unit tests exercise each step against hand-built frames; these run the two
CLI steps back to back over a fabricated but M5-shaped raw dataset, so a change
that breaks the *contract between* the steps (artifact names, column names, the
long -> wide reshape) fails here even when both steps pass in isolation.

Real Kaggle data is not needed: the fixture is a seeded, deterministic slice
long enough for baseline's default rolling-origin schedule
(season + step * (n_origins - 1) + max horizon = 7 + 49 + 28 = 84 days).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting import baseline, io, prepare_data

REPO_ROOT = Path(__file__).resolve().parent.parent

N_DAYS = 120
KEPT_SKUS = [f"FOODS_3_{i:03d}" for i in range(1, 5)]
DECOY_SKUS = ["FOODS_3_900", "HOBBIES_1_001"]


def _sales_row(item: str, cat: str, store: str, units: np.ndarray) -> dict[str, object]:
    return {
        "id": f"{item}_{store}_evaluation",
        "item_id": item,
        "dept_id": f"{cat}_3",
        "cat_id": cat,
        "store_id": store,
        "state_id": store[:2],
        **{f"d_{d}": int(u) for d, u in enumerate(units, start=1)},
    }


def write_raw_dataset(raw: Path) -> None:
    """An M5-shaped data/raw/ tree: in-scope SKUs, decoys, calendar and prices."""
    rng = np.random.default_rng(1234)
    weekly = np.array([3, 2, 2, 2, 4, 9, 8])  # weekday shape the seasonal naive can exploit

    rows = []
    for level, item in enumerate(KEPT_SKUS, start=1):
        # decreasing volume, so the top-N ranking has a stable, checkable order
        base = np.tile(weekly, N_DAYS // 7 + 1)[:N_DAYS] * (len(KEPT_SKUS) - level + 1)
        rows.append(_sales_row(item, "FOODS", "CA_1", base + rng.integers(0, 2, N_DAYS)))
    # out of scope: right category wrong store, and right store wrong category
    rows.append(_sales_row(DECOY_SKUS[0], "FOODS", "CA_2", np.full(N_DAYS, 99)))
    rows.append(_sales_row(DECOY_SKUS[1], "HOBBIES", "CA_1", np.full(N_DAYS, 99)))

    dates = pd.date_range("2013-01-01", periods=N_DAYS, freq="D")
    calendar = pd.DataFrame(
        {
            "d": [f"d_{d}" for d in range(1, N_DAYS + 1)],
            "date": dates.astype(str),
            "wm_yr_wk": 11101 + dates.isocalendar().week.to_numpy(),
            "wday": dates.dayofweek.to_numpy() + 1,
            "month": dates.month,
            "year": dates.year,
            "event_name_1": None,
            "event_type_1": None,
            "snap_CA": (dates.day <= 10).astype(int),
        }
    )

    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(raw / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(raw / "calendar.csv", index=False)
    pd.DataFrame(
        {
            "store_id": "CA_1",
            "item_id": KEPT_SKUS,
            "wm_yr_wk": 11101,
            "sell_price": np.arange(1.0, len(KEPT_SKUS) + 1.0),
        }
    ).to_csv(raw / "sell_prices.csv", index=False)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Point both steps at a throwaway data tree and hand back the raw fixture."""
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    monkeypatch.setattr(io, "RAW", raw)
    monkeypatch.setattr(io, "PROCESSED", processed)
    monkeypatch.setattr(prepare_data, "RAW", raw)
    monkeypatch.setattr(prepare_data, "PROCESSED", processed)
    monkeypatch.setitem(prepare_data.CONFIG, "n_skus", len(KEPT_SKUS))
    write_raw_dataset(raw)
    return raw, processed


def _run_module(module: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )


class TestPrepareDataToBaseline:
    def test_baseline_scores_every_sku_prepare_data_kept(self, pipeline, capsys):
        _, processed = pipeline

        prepare_data.main()
        baseline.main()
        capsys.readouterr()

        metrics = pd.read_csv(io.baseline_metrics_path(processed))
        assert sorted(metrics["item_id"].unique()) == KEPT_SKUS
        assert not set(metrics["item_id"]) & set(DECOY_SKUS)
        # one row per SKU x method x horizon
        assert len(metrics) == len(KEPT_SKUS) * 2 * len(baseline.HORIZONS)
        assert list(metrics.columns) == [
            "item_id",
            "method",
            "horizon",
            "rmse",
            "mape",
            "wape",
            "n_obs",
            "n_zero_actual",
        ]
        # every origin contributes one actual per forecast day
        for horizon in baseline.HORIZONS:
            at_horizon = metrics[metrics["horizon"] == horizon]
            assert (at_horizon["n_obs"] == baseline.N_ORIGINS * horizon).all()
        assert metrics[["rmse", "mape", "wape"]].notna().to_numpy().all()

    def test_seasonal_naive_beats_naive_on_a_weekly_series(self, pipeline, capsys):
        """The fixture is weekly-periodic, so the seasonal method must win."""
        _, processed = pipeline

        prepare_data.main()
        baseline.main()
        capsys.readouterr()

        wape = (
            pd.read_csv(io.baseline_metrics_path(processed))
            .groupby(["method", "horizon"])["wape"]
            .mean()
        )
        for horizon in baseline.HORIZONS:
            assert wape[("seasonal_naive", horizon)] < wape[("naive", horizon)]

    def test_parquet_reshapes_without_gaps(self, pipeline):
        _, processed = pipeline

        prepare_data.main()

        long = io.load_sales_long(processed)
        wide = io.load_sales_wide(processed=processed)
        assert len(long) == len(KEPT_SKUS) * N_DAYS
        assert wide.shape == (N_DAYS, len(KEPT_SKUS))
        assert not wide.isna().to_numpy().any()
        assert wide.index.is_monotonic_increasing
        # the SKU list is the parquet's SKUs, ranked by volume
        skus = io.skus_path(processed).read_text(encoding="utf-8").splitlines()
        assert skus == KEPT_SKUS
        assert set(skus) == set(long["item_id"].unique())

    def test_baseline_refuses_to_run_before_prepare_data(self, pipeline):
        with pytest.raises(FileNotFoundError, match="run `python -m forecasting.prepare_data`"):
            baseline.main()


class TestCliPipeline:
    def test_both_steps_succeed_from_the_working_directory(self, tmp_path):
        """`python -m forecasting.prepare_data && ... .baseline` on a fresh tree.

        Run with cwd=tmp_path because io.RAW / io.PROCESSED are relative paths.
        """
        write_raw_dataset(tmp_path / "data" / "raw")

        prep = _run_module("forecasting.prepare_data", tmp_path)
        assert prep.returncode == 0, prep.stderr
        assert f"Kept {len(KEPT_SKUS)} SKUs" in prep.stdout

        run = _run_module("forecasting.baseline", tmp_path)
        assert run.returncode == 0, run.stderr
        assert f"Backtesting {len(KEPT_SKUS)} SKUs" in run.stdout
        assert "Summary across SKUs" in run.stdout

        processed = tmp_path / "data" / "processed"
        assert {p.name for p in processed.iterdir()} == {
            io.SALES_LONG_NAME,
            io.SKUS_NAME,
            io.BASELINE_METRICS_NAME,
        }

    def test_metrics_are_reproducible_across_runs(self, tmp_path):
        write_raw_dataset(tmp_path / "data" / "raw")
        metrics_path = tmp_path / "data" / "processed" / io.BASELINE_METRICS_NAME

        outputs = []
        for _ in range(2):
            assert _run_module("forecasting.prepare_data", tmp_path).returncode == 0
            assert _run_module("forecasting.baseline", tmp_path).returncode == 0
            outputs.append(metrics_path.read_bytes())

        assert outputs[0] == outputs[1]
