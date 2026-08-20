"""Tests for the baseline forecasters' guard rails."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting import baseline, io
from forecasting.baseline import (
    compute_metrics,
    naive_forecast,
    rolling_origin_backtest,
    seasonal_naive_forecast,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _wide(n_days: int, n_items: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.integers(0, 10, size=(n_days, n_items)).astype(float),
        index=dates,
        columns=[f"ITEM_{i}" for i in range(n_items)],
    )


def test_naive_forecast_rejects_empty_history():
    with pytest.raises(ValueError, match="at least one observation"):
        naive_forecast(np.array([]), 7)


def test_naive_forecast_rejects_nan_last_value():
    with pytest.raises(ValueError, match="cannot forecast from a gap"):
        naive_forecast(np.array([1.0, np.nan]), 7)


def test_seasonal_naive_forecast_rejects_short_history():
    with pytest.raises(ValueError, match="at least season=7"):
        seasonal_naive_forecast(np.arange(5, dtype=float), 28)


def test_seasonal_naive_forecast_fills_horizon():
    out = seasonal_naive_forecast(np.arange(14, dtype=float), 28)
    assert out.shape == (28,)
    assert out[0] == 7.0


def test_backtest_rejects_missing_cells():
    wide = _wide(200)
    wide.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="missing .* cell"):
        rolling_origin_backtest(wide)


def test_backtest_rejects_insufficient_history():
    with pytest.raises(ValueError, match="Not enough history"):
        rolling_origin_backtest(_wide(40))


def test_backtest_and_metrics_roundtrip():
    results = rolling_origin_backtest(_wide(200))
    assert set(results["day_offset"]) == set(range(1, 29))
    metrics = compute_metrics(results)
    assert len(metrics) == 2 * 2 * 2  # horizons x methods x items
    assert metrics["rmse"].notna().all()


def test_compute_metrics_rejects_empty_results():
    with pytest.raises(ValueError, match="No backtest results"):
        compute_metrics(pd.DataFrame(columns=["item_id", "day_offset", "actual"]))


# --- Numeric behavior of the forecasters and metrics --------------------------


def _ramp_wide(n_days: int, items=("ITEM_1", "ITEM_2")) -> pd.DataFrame:
    """Deterministic increasing series, so expected forecasts are readable."""
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    data = {item: np.arange(n_days, dtype=float) + offset for offset, item in enumerate(items)}
    return pd.DataFrame(data, index=dates)


class TestNaiveForecast:
    def test_repeats_last_observation(self):
        forecast = naive_forecast(np.array([1, 2, 5]), 4)
        assert forecast.tolist() == [5.0, 5.0, 5.0, 5.0]
        assert forecast.dtype == float

    @pytest.mark.parametrize("horizon", [0, -1])
    def test_rejects_non_positive_horizon(self, horizon):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            naive_forecast(np.array([3.0]), horizon)


class TestSeasonalNaiveForecast:
    def test_tiles_last_season(self):
        history = np.arange(1, 15, dtype=float)  # season 7 -> last week is 8..14
        forecast = seasonal_naive_forecast(history, 10)
        assert forecast.tolist() == [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]

    def test_horizon_shorter_than_season_truncates(self):
        assert seasonal_naive_forecast(np.arange(1, 15, dtype=float), 3).tolist() == [8, 9, 10]

    def test_custom_season(self):
        forecast = seasonal_naive_forecast(np.array([1.0, 2.0, 3.0, 4.0]), 5, season=2)
        assert forecast.tolist() == [3, 4, 3, 4, 3]

    @pytest.mark.parametrize("season", [0, -7])
    def test_rejects_non_positive_season(self, season):
        with pytest.raises(ValueError, match="season must be >= 1"):
            seasonal_naive_forecast(np.arange(10, dtype=float), 7, season=season)


class TestRollingOriginBacktest:
    def test_shape_and_columns(self):
        results = rolling_origin_backtest(
            _ramp_wide(40), horizons=(3, 7), season=7, step=7, n_origins=2
        )
        assert list(results.columns) == [
            "item_id",
            "origin_date",
            "day_offset",
            "actual",
            "naive_pred",
            "seasonal_naive_pred",
        ]
        # 2 origins x 2 items x max horizon of 7 days
        assert len(results) == 2 * 2 * 7
        assert sorted(results["day_offset"].unique()) == [1, 2, 3, 4, 5, 6, 7]

    def test_origins_are_step_spaced_and_end_before_the_last_horizon(self):
        wide = _ramp_wide(40)
        results = rolling_origin_backtest(wide, horizons=(7,), season=7, step=7, n_origins=3)
        assert sorted(results["origin_date"].unique()) == list(wide.index[[18, 25, 32]])

    def test_predictions_and_actuals_are_aligned(self):
        wide = _ramp_wide(40, items=("ITEM_1",))
        results = rolling_origin_backtest(wide, horizons=(7,), season=7, step=7, n_origins=1)
        origin_idx = 32  # len - 1 - max_h
        expected_actual = wide["ITEM_1"].to_numpy()[origin_idx + 1 : origin_idx + 8]
        assert results["actual"].tolist() == expected_actual.tolist()
        # naive: flat at the last observed value
        assert results["naive_pred"].unique().tolist() == [wide["ITEM_1"].iloc[origin_idx]]
        # seasonal naive: the week preceding (and including) the origin
        assert results["seasonal_naive_pred"].tolist() == (
            wide["ITEM_1"].to_numpy()[origin_idx - 6 : origin_idx + 1].tolist()
        )

    def test_empty_horizons_rejected(self):
        with pytest.raises(ValueError, match="horizons must not be empty"):
            rolling_origin_backtest(_ramp_wide(200), horizons=())

    def test_empty_frame_rejected(self):
        with pytest.raises(ValueError, match="wide frame is empty"):
            rolling_origin_backtest(pd.DataFrame())


def _results_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["item_id", "day_offset", "actual", "naive_pred", "seasonal_naive_pred"]
    )


class TestComputeMetrics:
    def test_metrics_for_a_single_item(self):
        results = _results_frame([("A", 1, 10.0, 8.0, 12.0), ("A", 2, 20.0, 8.0, 12.0)])
        naive = compute_metrics(results, horizons=(2,)).query("method == 'naive'").iloc[0]
        assert naive["rmse"] == pytest.approx(np.sqrt((2**2 + 12**2) / 2))
        assert naive["mape"] == pytest.approx((2 / 10 + 12 / 20) / 2 * 100)
        assert naive["wape"] == pytest.approx((2 + 12) / 30 * 100)
        assert naive["n_obs"] == 2
        assert naive["n_zero_actual"] == 0

    def test_method_suffix_is_stripped_and_rows_cover_every_horizon(self):
        results = _results_frame(
            [(item, offset, 5.0, 4.0, 6.0) for item in ("A", "B") for offset in range(1, 5)]
        )
        metrics = compute_metrics(results, horizons=(2, 4))
        assert set(metrics["method"]) == {"naive", "seasonal_naive"}
        assert len(metrics) == 2 * 2 * 2  # horizons x methods x items
        assert metrics[metrics.horizon == 2]["n_obs"].tolist() == [2, 2, 2, 2]

    def test_horizon_cutoff_excludes_later_day_offsets(self):
        results = _results_frame([("A", 1, 10.0, 10.0, 10.0), ("A", 5, 10.0, 0.0, 0.0)])
        metrics = compute_metrics(results, horizons=(1,))
        assert metrics["rmse"].tolist() == [0.0, 0.0]
        assert metrics["n_obs"].tolist() == [1, 1]

    def test_zero_actuals_are_excluded_from_mape_and_counted(self):
        results = _results_frame([("A", 1, 0.0, 2.0, 2.0), ("A", 2, 4.0, 2.0, 2.0)])
        naive = compute_metrics(results, horizons=(2,)).iloc[0]
        assert naive["mape"] == pytest.approx(50.0)
        assert naive["n_zero_actual"] == 1
        assert naive["wape"] == pytest.approx((2 + 2) / 4 * 100)

    def test_all_zero_actuals_give_nan_mape_and_wape(self):
        naive = compute_metrics(_results_frame([("A", 1, 0.0, 1.0, 1.0)]), horizons=(1,)).iloc[0]
        assert np.isnan(naive["mape"])
        assert np.isnan(naive["wape"])
        assert naive["n_zero_actual"] == 1

    def test_perfect_forecast_scores_zero(self):
        metrics = compute_metrics(_results_frame([("A", 1, 7.0, 7.0, 7.0)]), horizons=(1,))
        assert metrics["rmse"].tolist() == [0.0, 0.0]
        assert metrics["mape"].tolist() == [0.0, 0.0]
        assert metrics["wape"].tolist() == [0.0, 0.0]


class TestMain:
    def test_writes_metrics_csv(self, tmp_path, monkeypatch, capsys):
        n_days = 120
        dates = pd.date_range("2014-01-01", periods=n_days, freq="D")
        units = np.arange(n_days, dtype=float)
        long_df = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "item_id": ["A"] * n_days + ["B"] * n_days,
                "units": list(units) + list(units * 2),
            }
        )
        processed = tmp_path / "processed"
        processed.mkdir()
        long_df.to_parquet(processed / "sales_long.parquet", index=False)
        monkeypatch.setattr(io, "PROCESSED", processed)

        baseline.main()

        written = pd.read_csv(processed / "baseline_metrics.csv")
        assert set(written["item_id"]) == {"A", "B"}
        assert set(written["method"]) == {"naive", "seasonal_naive"}
        assert sorted(written["horizon"].unique()) == list(baseline.HORIZONS)
        out = capsys.readouterr().out
        assert "Backtesting 2 SKUs" in out
        assert "Summary across SKUs" in out

    def test_missing_parquet_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(io, "PROCESSED", tmp_path)
        with pytest.raises(FileNotFoundError, match="run `python -m forecasting.prepare_data`"):
            baseline.main()

    def test_missing_columns_raise(self, tmp_path, monkeypatch):
        pd.DataFrame({"date": [1], "units": [2]}).to_parquet(tmp_path / "sales_long.parquet")
        monkeypatch.setattr(io, "PROCESSED", tmp_path)
        with pytest.raises(ValueError, match=r"missing column\(s\) \['item_id'\]"):
            baseline.main()

    def test_duplicate_date_item_pairs_raise(self, tmp_path, monkeypatch):
        pd.DataFrame(
            {"date": ["2015-01-01"] * 2, "item_id": ["A", "A"], "units": [1.0, 2.0]}
        ).to_parquet(tmp_path / "sales_long.parquet")
        monkeypatch.setattr(io, "PROCESSED", tmp_path)
        with pytest.raises(ValueError, match="duplicate"):
            baseline.main()


class TestGuardRailMessages:
    def test_seasonal_naive_rejects_non_positive_horizon(self):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            seasonal_naive_forecast(np.arange(10, dtype=float), 0)

    def test_seasonal_naive_rejects_nan_inside_the_season_window(self):
        history = np.arange(14, dtype=float)
        history[-3] = np.nan
        with pytest.raises(ValueError, match="cannot forecast from a gap"):
            seasonal_naive_forecast(history, 7)

    def test_compute_metrics_rejects_horizon_with_no_rows(self):
        results = _results_frame([("A", 5, 1.0, 1.0, 1.0)])
        with pytest.raises(ValueError, match="No backtest rows within horizon 1"):
            compute_metrics(results, horizons=(1,))

    def test_compute_metrics_rejects_non_finite_values(self):
        results = _results_frame([("A", 1, np.nan, 1.0, 1.0)])
        with pytest.raises(ValueError, match="Non-finite actuals"):
            compute_metrics(results, horizons=(1,))


def test_module_entrypoint_exits_nonzero_with_a_readable_error(tmp_path):
    """`python -m forecasting.baseline` maps the guard rails onto exit code 1."""
    proc = subprocess.run(
        [sys.executable, "-m", "forecasting.baseline"],
        cwd=tmp_path,  # no data/processed here
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 1
    assert proc.stderr.startswith("ERROR: ")
    assert "prepare_data" in proc.stderr
