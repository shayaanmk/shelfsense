"""Tests for the baseline forecasters' guard rails."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.baseline import (
    compute_metrics,
    naive_forecast,
    rolling_origin_backtest,
    seasonal_naive_forecast,
)


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
