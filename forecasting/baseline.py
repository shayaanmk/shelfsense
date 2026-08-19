"""Naive and seasonal-naive forecasting baselines (build plan step 2).

These exist to give the LightGBM model (step 2b) something concrete to beat.
Two methods:
  - naive:          forecast = last observed value, flat across the horizon.
  - seasonal_naive:  forecast = value from the same weekday one week back,
                      tiled across the horizon (period = 7 days).

Validated with rolling-origin backtesting (step 3): multiple forecast origins
spaced a week apart, each scored at both the 7-day and 28-day horizon,
per SKU.

Run:  python -m forecasting.baseline
Output: data/processed/baseline_metrics.csv (per-SKU, per-method, per-horizon)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
HORIZONS = (7, 28)
SEASON = 7          # weekly seasonality
STEP = 7             # spacing between rolling origins, in days
N_ORIGINS = 8         # number of backtest origins (8 weeks of rolling-origin coverage)


def naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.full(horizon, history[-1], dtype=float)


def seasonal_naive_forecast(history: np.ndarray, horizon: int, season: int = SEASON) -> np.ndarray:
    last_season = history[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon].astype(float)


def rolling_origin_backtest(
    wide: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    season: int = SEASON,
    step: int = STEP,
    n_origins: int = N_ORIGINS,
) -> pd.DataFrame:
    """For each origin and SKU, forecast max(horizons) days ahead and record
    actual vs. predicted for every day of the horizon. compute_metrics()
    then slices by day_offset to score the 7-day and 28-day cutoffs."""
    dates = wide.index
    max_h = max(horizons)

    last_origin_idx = len(dates) - 1 - max_h
    if last_origin_idx - step * (n_origins - 1) < season:
        raise ValueError("Not enough history for the requested origins/horizon/season.")
    origin_idxs = [last_origin_idx - step * i for i in range(n_origins)]

    records = []
    for idx in origin_idxs:
        origin_date = dates[idx]
        for item in wide.columns:
            hist = wide[item].to_numpy()[: idx + 1]
            future_actual = wide[item].to_numpy()[idx + 1 : idx + 1 + max_h]
            naive_pred = naive_forecast(hist, max_h)
            sn_pred = seasonal_naive_forecast(hist, max_h, season)
            for day_offset in range(max_h):
                records.append(
                    {
                        "item_id": item,
                        "origin_date": origin_date,
                        "day_offset": day_offset + 1,
                        "actual": future_actual[day_offset],
                        "naive_pred": naive_pred[day_offset],
                        "seasonal_naive_pred": sn_pred[day_offset],
                    }
                )
    return pd.DataFrame(records)


def compute_metrics(results: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Per-SKU RMSE/MAPE/WAPE for each method at each horizon cutoff.

    MAPE is undefined at actual=0, which is common in M5 (intermittent
    demand) -- those rows are excluded and counted in n_zero_actual so the
    exclusion is visible rather than silently skewing the metric. WAPE
    (sum of errors / sum of actuals) doesn't have this problem and is
    reported alongside as the more robust number for these SKUs.
    """
    rows = []
    for horizon in horizons:
        subset = results[results["day_offset"] <= horizon]
        for method in ("naive_pred", "seasonal_naive_pred"):
            for item_id, g in subset.groupby("item_id"):
                actual = g["actual"].to_numpy()
                pred = g[method].to_numpy()
                err = actual - pred
                nonzero = actual != 0

                rmse = float(np.sqrt(np.mean(err**2)))
                mape = (
                    float(np.mean(np.abs(err[nonzero]) / actual[nonzero]) * 100)
                    if nonzero.any()
                    else float("nan")
                )
                wape = float(np.sum(np.abs(err)) / np.sum(actual) * 100) if actual.sum() > 0 else float("nan")

                rows.append(
                    {
                        "item_id": item_id,
                        "method": method.removesuffix("_pred"),
                        "horizon": horizon,
                        "rmse": rmse,
                        "mape": mape,
                        "wape": wape,
                        "n_obs": len(g),
                        "n_zero_actual": int((~nonzero).sum()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    long_df = pd.read_parquet(PROCESSED / "sales_long.parquet")
    wide = long_df.pivot(index="date", columns="item_id", values="units").sort_index()

    print(f"Backtesting {wide.shape[1]} SKUs over {N_ORIGINS} rolling origins "
          f"(step={STEP}d, horizons={HORIZONS})...")
    results = rolling_origin_backtest(wide)
    metrics = compute_metrics(results)

    out = PROCESSED / "baseline_metrics.csv"
    metrics.to_csv(out, index=False)
    print(f"Wrote {out}\n")

    summary = (
        metrics.groupby(["method", "horizon"])[["rmse", "mape", "wape"]]
        .agg(["mean", "median"])
        .round(2)
    )
    print("=== Summary across SKUs ===")
    print(summary, "\n")

    for horizon in HORIZONS:
        sn = metrics[(metrics.method == "seasonal_naive") & (metrics.horizon == horizon)]
        worst = sn.nlargest(5, "wape")[["item_id", "rmse", "mape", "wape"]]
        print(f"=== Worst 5 SKUs, seasonal_naive @ {horizon}d (by WAPE) ===")
        print(worst.to_string(index=False), "\n")


if __name__ == "__main__":
    main()
