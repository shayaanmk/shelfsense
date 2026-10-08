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

import sys

import numpy as np
import pandas as pd

from forecasting import io

HORIZONS = (7, 28)
SEASON = 7          # weekly seasonality
STEP = 7             # spacing between rolling origins, in days
N_ORIGINS = 8         # number of backtest origins (8 weeks of rolling-origin coverage)


def naive_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if history.size == 0:
        raise ValueError("naive_forecast needs at least one observation.")
    last = history[-1]
    if not np.isfinite(last):
        raise ValueError(f"last observation is {last}; cannot forecast from a gap.")
    return np.full(horizon, last, dtype=float)


def seasonal_naive_forecast(history: np.ndarray, horizon: int, season: int = SEASON) -> np.ndarray:
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if season < 1:
        raise ValueError(f"season must be >= 1, got {season}.")
    if history.size < season:
        raise ValueError(
            f"seasonal_naive_forecast needs at least season={season} observations, "
            f"got {history.size}; a shorter history would yield a truncated forecast."
        )
    last_season = history[-season:]
    if not np.isfinite(last_season).all():
        raise ValueError(
            f"last {season} observations contain non-finite values; cannot forecast from a gap."
        )
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon].astype(float)


def origin_indices(n_dates: int, max_horizon: int, step: int, n_origins: int, min_history: int) -> list[int]:
    """Rolling-origin indices ending as late as the data allows.

    Shared by rolling_origin_backtest and forecasting.lgbm_model's time split
    so both draw origins from the identical index math -- the LightGBM
    model's test set lines up exactly with this module's, SKU for SKU and
    horizon for horizon.
    """
    last_origin_idx = n_dates - 1 - max_horizon
    first_origin_idx = last_origin_idx - step * (n_origins - 1)
    if first_origin_idx < min_history - 1:
        raise ValueError(
            f"Not enough history: {n_dates} days cannot cover {n_origins} origins "
            f"(step={step}) at horizon {max_horizon} with min_history {min_history}; "
            f"need at least {min_history + step * (n_origins - 1) + max_horizon} days."
        )
    return [last_origin_idx - step * i for i in range(n_origins)]


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
    if not horizons:
        raise ValueError("horizons must not be empty.")
    if wide.empty:
        raise ValueError("No sales data to backtest: the wide frame is empty.")
    if wide.isna().to_numpy().any():
        n_missing = int(wide.isna().to_numpy().sum())
        gappy = wide.columns[wide.isna().any()].tolist()
        raise ValueError(
            f"{n_missing} missing (date, SKU) cell(s) across {len(gappy)} SKU(s) "
            f"(e.g. {gappy[:5]}). Forecasts and metrics over these would silently be NaN; "
            "fix the upstream data (re-run forecasting.prepare_data) first."
        )

    dates = wide.index
    max_h = max(horizons)
    origin_idxs = origin_indices(len(dates), max_h, step, n_origins, season)

    records = []
    for idx in origin_idxs:
        origin_date = dates[idx]
        for item in wide.columns:
            hist = wide[item].to_numpy()[: idx + 1]
            future_actual = wide[item].to_numpy()[idx + 1 : idx + 1 + max_h]
            if future_actual.size != max_h:
                raise ValueError(
                    f"Origin {origin_date} leaves only {future_actual.size} of {max_h} "
                    f"actual days for {item}."
                )
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


def compute_metrics(
    results: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    pred_cols: tuple[str, ...] = ("naive_pred", "seasonal_naive_pred"),
) -> pd.DataFrame:
    """Per-SKU RMSE/MAPE/WAPE for each method at each horizon cutoff.

    `pred_cols` names the prediction column(s) in `results` to score; each
    must end in "_pred" (the method name reported is that column with the
    suffix stripped). Defaults to this module's two baselines, but any model
    that produces a (item_id, day_offset, actual, <method>_pred) table --
    forecasting.lgbm_model included -- can reuse this rather than
    reimplementing the MAPE/WAPE zero-handling.

    MAPE is undefined at actual=0, which is common in M5 (intermittent
    demand) -- those rows are excluded and counted in n_zero_actual so the
    exclusion is visible rather than silently skewing the metric. WAPE
    (sum of errors / sum of actuals) doesn't have this problem and is
    reported alongside as the more robust number for these SKUs.
    """
    if results.empty:
        raise ValueError("No backtest results to score.")

    rows = []
    for horizon in horizons:
        subset = results[results["day_offset"] <= horizon]
        if subset.empty:
            raise ValueError(f"No backtest rows within horizon {horizon}.")
        for method in pred_cols:
            for item_id, g in subset.groupby("item_id"):
                actual = g["actual"].to_numpy(dtype=float)
                pred = g[method].to_numpy(dtype=float)
                if not (np.isfinite(actual).all() and np.isfinite(pred).all()):
                    raise ValueError(
                        f"Non-finite actuals or {method} values for {item_id} "
                        f"at horizon {horizon}; metrics would be NaN."
                    )
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
    wide = io.load_sales_wide()

    print(f"Backtesting {wide.shape[1]} SKUs over {N_ORIGINS} rolling origins "
          f"(step={STEP}d, horizons={HORIZONS})...")
    results = rolling_origin_backtest(wide)
    metrics = compute_metrics(results)

    io.ensure_processed()
    out = io.baseline_metrics_path()
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
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
