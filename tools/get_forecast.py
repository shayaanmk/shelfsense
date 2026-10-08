"""get_forecast(sku, horizon): real forecast values for one of the agent's
tools.

Serves predictions from the trained LightGBM model (forecasting/lgbm_model.py)
through the same feature pipeline it was trained with. Falls back to the
seasonal-naive baseline when the trained model file isn't present yet (e.g. a
fresh checkout before `python -m forecasting.lgbm_model` has been run) --
never fabricates a number.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting import baseline, features, io, lgbm_model
from forecasting.prepare_data import CONFIG as SCOPE_CONFIG


def get_forecast(sku: str, horizon: int, as_of: str | pd.Timestamp | None = None) -> dict:
    """`as_of` defaults to the real last sales date (the copilot's "today").
    Pass an earlier date to forecast from THAT point instead -- e.g.
    agent.restock_agent deciding on a date within commerce.pos's 78-day
    replay window needs the model's view from that simulated "today", not
    from the dataset's actual final day."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")

    wide = io.load_sales_wide()
    if sku not in wide.columns:
        raise ValueError(f"Unknown SKU {sku!r}; {wide.shape[1]} SKUs are in scope, see data/processed/skus.txt.")

    if as_of is None:
        origin_date = wide.index[-1]
    else:
        origin_date = pd.Timestamp(as_of)
        if origin_date not in wide.index:
            raise ValueError(f"as_of={origin_date.date()} is not a known sales date for {sku}.")

    model_path = io.PROCESSED / "lgbm_model.txt"

    if model_path.exists():
        forecast = _lgbm_forecast(wide, sku, origin_date, horizon, model_path)
        model_used = "lgbm"
    else:
        preds = baseline.seasonal_naive_forecast(wide[sku].to_numpy(), horizon)
        dates = origin_date + pd.to_timedelta(np.arange(1, horizon + 1), unit="D")
        forecast = list(zip(dates, preds))
        model_used = "seasonal_naive"

    return {
        "sku": sku,
        "as_of_date": str(origin_date.date()),
        "horizon": horizon,
        "model": model_used,
        "forecast": [
            {"date": str(d.date()), "predicted_units": round(float(p), 2)} for d, p in forecast
        ],
    }


def _lgbm_forecast(
    wide: pd.DataFrame, sku: str, origin_date: pd.Timestamp, horizon: int, model_path
) -> list[tuple[pd.Timestamp, float]]:
    long_df = io.load_sales_long()
    # historical-only calendar: must match what category_dtypes was fit on at training
    # time, or a category could get a different integer code than the trained booster expects.
    train_calendar = lgbm_model.load_calendar(long_df)
    # full-range calendar: the only one with rows past the last sales day, needed to
    # look up target_wday/month/event/snap *values* for the dates being forecast.
    future_calendar = io.load_future_calendar(SCOPE_CONFIG["store"])

    dtypes = lgbm_model.category_dtypes(wide.columns, train_calendar)
    table = features.build_feature_table(
        wide, future_calendar, pd.DatetimeIndex([origin_date]), range(1, horizon + 1), include_target=False
    )
    table = lgbm_model.apply_category_dtypes(table, dtypes)
    table = table[table["item_id"] == sku].sort_values("horizon")

    booster = lgb.Booster(model_file=str(model_path))
    preds = np.clip(booster.predict(table[lgbm_model.FEATURE_COLS]), 0, None)
    dates = origin_date + pd.to_timedelta(table["horizon"].to_numpy(), unit="D")
    return list(zip(dates, preds))
