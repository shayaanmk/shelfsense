"""detect_anomalies(sku, lookback_days): forecast-residual anomaly detection
for one of the agent's tools.

Demand-side: flags days where actual sales deviate from the seasonal-naive
expectation (same weekday, one week prior) by more than Z_THRESHOLD trailing
standard deviations of that residual -- so "anomalous" means "surprising
relative to this SKU's own recent week-over-week variability," not a fixed
unit threshold that would be meaningless across a 50-SKU range spanning
near-zero to double-digit daily volume.

Supply-side: flags stockout days straight from the inventory simulation
(tools/inventory_sim.py) -- a stockout is unambiguous, no threshold needed.

Z_THRESHOLD is a first-pass default (own it, per CLAUDE.md), meant to be
tuned once eval/ can measure precision/recall against injected synthetic
anomalies (build plan step 6) rather than guessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting import io

Z_THRESHOLD = 2.5
RESIDUAL_STD_WINDOW = 90  # trailing days used to estimate "normal" week-over-week variability


def _residual_zscores(units: pd.Series) -> pd.Series:
    residual = units - units.shift(7)
    trailing_std = residual.shift(1).rolling(RESIDUAL_STD_WINDOW, min_periods=14).std()
    return residual / trailing_std.replace(0, np.nan)


def detect_anomalies(
    sku: str,
    lookback_days: int = 28,
    wide: pd.DataFrame | None = None,
    inventory: pd.DataFrame | None = None,
    z_threshold: float = Z_THRESHOLD,
) -> dict:
    """`wide`/`inventory` default to disk (io.load_sales_wide/load_inventory)
    -- the real tool call path. eval.anomaly_eval passes injected in-memory
    frames instead, so a synthetic scenario can be scored without writing it
    to data/processed/ first. `z_threshold` defaults to the module constant
    but is overridable so eval.anomaly_eval can sweep it without touching
    the production default."""
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}.")

    wide = io.load_sales_wide() if wide is None else wide
    if sku not in wide.columns:
        raise ValueError(f"Unknown SKU {sku!r}; {wide.shape[1]} SKUs are in scope, see data/processed/skus.txt.")
    if lookback_days > len(wide.index):
        raise ValueError(f"Only {len(wide.index)} days of history exist; requested lookback {lookback_days}.")

    units = wide[sku]
    z = _residual_zscores(units)
    same_weekday_prior = units.shift(7)
    window = units.index[-lookback_days:]

    anomalies = []
    for date in window:
        score = z.loc[date]
        if pd.notna(score) and abs(score) >= z_threshold:
            anomalies.append(
                {
                    "date": str(date.date()),
                    "type": "demand_spike" if score > 0 else "demand_drop",
                    "severity": round(float(abs(score)), 2),
                    "evidence": {
                        "actual_units": float(units.loc[date]),
                        "same_weekday_prior_week": float(same_weekday_prior.loc[date]),
                        "z_score": round(float(score), 2),
                    },
                }
            )

    try:
        inv = io.load_inventory() if inventory is None else inventory
        item_inv = inv[(inv["item_id"] == sku) & (inv["date"].isin(window))]
        for _, row in item_inv[item_inv["stockout"]].iterrows():
            anomalies.append(
                {
                    "date": str(row["date"].date()),
                    "type": "stockout",
                    "severity": None,
                    "evidence": {
                        "on_hand": float(row["on_hand"]),
                        "demand": float(row["demand"]),
                        "unmet_demand": float(row["unmet_demand"]),
                    },
                }
            )
    except FileNotFoundError:
        pass  # inventory simulation hasn't been run yet; demand-side detection still works

    anomalies.sort(key=lambda a: a["date"])
    return {
        "sku": sku,
        "as_of_date": str(window[-1].date()),
        "lookback_days": lookback_days,
        "z_threshold": z_threshold,
        "anomalies": anomalies,
    }
