import numpy as np
import pandas as pd
import pytest

from forecasting import baseline


def make_wide(n_days: int, items=("ITEM_1", "ITEM_2")) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    data = {item: np.arange(n_days, dtype=float) + offset for offset, item in enumerate(items)}
    return pd.DataFrame(data, index=dates)


class TestNaiveForecast:
    def test_repeats_last_observation(self):
        forecast = baseline.naive_forecast(np.array([1, 2, 5]), 4)
        assert forecast.tolist() == [5.0, 5.0, 5.0, 5.0]
        assert forecast.dtype == float

    def test_zero_horizon(self):
        assert baseline.naive_forecast(np.array([3]), 0).tolist() == []


class TestSeasonalNaiveForecast:
    def test_tiles_last_season(self):
        history = np.arange(1, 15)  # 1..14, season 7 -> last week is 8..14
        forecast = baseline.seasonal_naive_forecast(history, 10)
        assert forecast.tolist() == [8, 9, 10, 11, 12, 13, 14, 8, 9, 10]

    def test_horizon_shorter_than_season_truncates(self):
        forecast = baseline.seasonal_naive_forecast(np.arange(1, 15), 3)
        assert forecast.tolist() == [8, 9, 10]

    def test_custom_season(self):
        forecast = baseline.seasonal_naive_forecast(np.array([1, 2, 3, 4]), 5, season=2)
        assert forecast.tolist() == [3, 4, 3, 4, 3]

    def test_history_shorter_than_season_returns_a_short_forecast(self):
        # reps is derived from horizon/season, so a history shorter than `season`
        # yields fewer than `horizon` values instead of tiling to fill it.
        forecast = baseline.seasonal_naive_forecast(np.array([2.0, 4.0]), 4, season=7)
        assert forecast.tolist() == [2, 4]


class TestRollingOriginBacktest:
    def test_shape_and_columns(self):
        wide = make_wide(40)
        results = baseline.rolling_origin_backtest(
            wide, horizons=(3, 7), season=7, step=7, n_origins=2
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
        wide = make_wide(40)
        results = baseline.rolling_origin_backtest(
            wide, horizons=(7,), season=7, step=7, n_origins=3
        )
        origins = sorted(results["origin_date"].unique())
        assert origins == list(wide.index[[18, 25, 32]])

    def test_predictions_and_actuals_are_aligned(self):
        wide = make_wide(40, items=("ITEM_1",))
        results = baseline.rolling_origin_backtest(
            wide, horizons=(7,), season=7, step=7, n_origins=1
        )
        origin_idx = 32  # len - 1 - max_h
        expected_actual = wide["ITEM_1"].to_numpy()[origin_idx + 1 : origin_idx + 8]
        assert results["actual"].tolist() == expected_actual.tolist()
        # naive: flat at the last observed value
        assert results["naive_pred"].unique().tolist() == [wide["ITEM_1"].iloc[origin_idx]]
        # seasonal naive: the week preceding (and including) the origin
        assert results["seasonal_naive_pred"].tolist() == (
            wide["ITEM_1"].to_numpy()[origin_idx - 6 : origin_idx + 1].tolist()
        )

    def test_insufficient_history_raises(self):
        with pytest.raises(ValueError, match="Not enough history"):
            baseline.rolling_origin_backtest(
                make_wide(20), horizons=(7,), season=7, step=7, n_origins=3
            )

    def test_defaults_need_a_long_series(self):
        with pytest.raises(ValueError, match="Not enough history"):
            baseline.rolling_origin_backtest(make_wide(60))


def results_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["item_id", "day_offset", "actual", "naive_pred", "seasonal_naive_pred"]
    )


class TestComputeMetrics:
    def test_metrics_for_a_single_item(self):
        results = results_frame(
            [
                ("A", 1, 10.0, 8.0, 12.0),
                ("A", 2, 20.0, 8.0, 12.0),
            ]
        )
        metrics = baseline.compute_metrics(results, horizons=(2,))
        naive = metrics[metrics.method == "naive"].iloc[0]
        assert naive["rmse"] == pytest.approx(np.sqrt((2**2 + 12**2) / 2))
        assert naive["mape"] == pytest.approx((2 / 10 + 12 / 20) / 2 * 100)
        assert naive["wape"] == pytest.approx((2 + 12) / 30 * 100)
        assert naive["n_obs"] == 2
        assert naive["n_zero_actual"] == 0
        assert set(metrics["method"]) == {"naive", "seasonal_naive"}

    def test_method_suffix_is_stripped_and_rows_cover_every_horizon(self):
        results = results_frame(
            [("A", offset, 5.0, 4.0, 6.0) for offset in range(1, 5)]
            + [("B", offset, 5.0, 4.0, 6.0) for offset in range(1, 5)]
        )
        metrics = baseline.compute_metrics(results, horizons=(2, 4))
        assert len(metrics) == 2 * 2 * 2  # horizons x methods x items
        assert sorted(metrics["horizon"].unique()) == [2, 4]
        assert metrics[metrics.horizon == 2]["n_obs"].tolist() == [2, 2, 2, 2]

    def test_horizon_cutoff_excludes_later_day_offsets(self):
        results = results_frame(
            [
                ("A", 1, 10.0, 10.0, 10.0),
                ("A", 5, 10.0, 0.0, 0.0),
            ]
        )
        metrics = baseline.compute_metrics(results, horizons=(1,))
        assert metrics["rmse"].tolist() == [0.0, 0.0]
        assert metrics["n_obs"].tolist() == [1, 1]

    def test_zero_actuals_are_excluded_from_mape_and_counted(self):
        results = results_frame(
            [
                ("A", 1, 0.0, 2.0, 2.0),
                ("A", 2, 4.0, 2.0, 2.0),
            ]
        )
        naive = baseline.compute_metrics(results, horizons=(2,)).iloc[0]
        assert naive["mape"] == pytest.approx(50.0)
        assert naive["n_zero_actual"] == 1
        assert naive["wape"] == pytest.approx((2 + 2) / 4 * 100)

    def test_all_zero_actuals_give_nan_mape_and_wape(self):
        results = results_frame([("A", 1, 0.0, 1.0, 1.0)])
        naive = baseline.compute_metrics(results, horizons=(1,)).iloc[0]
        assert np.isnan(naive["mape"])
        assert np.isnan(naive["wape"])
        assert naive["n_zero_actual"] == 1

    def test_perfect_forecast_scores_zero(self):
        results = results_frame([("A", 1, 7.0, 7.0, 7.0)])
        metrics = baseline.compute_metrics(results, horizons=(1,))
        assert metrics["rmse"].tolist() == [0.0, 0.0]
        assert metrics["mape"].tolist() == [0.0, 0.0]
        assert metrics["wape"].tolist() == [0.0, 0.0]


class TestMain:
    def test_writes_metrics_csv(self, tmp_path, monkeypatch, capsys):
        n_days = 120
        dates = pd.date_range("2014-01-01", periods=n_days, freq="D")
        long_df = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "item_id": ["A"] * n_days + ["B"] * n_days,
                "units": list(np.arange(n_days, dtype=float))
                + list(np.arange(n_days, dtype=float) * 2),
            }
        )
        processed = tmp_path / "processed"
        processed.mkdir()
        long_df.to_parquet(processed / "sales_long.parquet", index=False)
        monkeypatch.setattr(baseline, "PROCESSED", processed)

        baseline.main()

        written = pd.read_csv(processed / "baseline_metrics.csv")
        assert set(written["item_id"]) == {"A", "B"}
        assert set(written["method"]) == {"naive", "seasonal_naive"}
        assert sorted(written["horizon"].unique()) == list(baseline.HORIZONS)
        out = capsys.readouterr().out
        assert "Backtesting 2 SKUs" in out
        assert "Summary across SKUs" in out
