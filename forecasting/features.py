"""Lag / rolling / calendar feature engineering for the LightGBM demand model.

Every feature is computed "as of" an origin date, using only units on or
before that date. The target for a (item, origin, horizon) row is the unit
count `horizon` days after origin. Because features never look past origin,
a row is leakage-free by construction regardless of which time split
(train/val/test) it ends up in -- see forecasting.lgbm_model.time_split.

Calendar features (day-of-week, month, event, SNAP) describe the *target*
date, not the origin -- legitimate because the M5 calendar is known in
advance (public holidays, promo events, SNAP schedule), not derived from
sales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (7, 14, 21, 28)
ROLLING_WINDOWS = (7, 28)
MIN_HISTORY = max(LAGS)  # an origin needs this many prior days for lag_28

CALENDAR_COLS = ["wday", "month", "event_type_1", "snap"]


def build_feature_table(
    wide: pd.DataFrame,
    calendar: pd.DataFrame,
    origin_dates: pd.DatetimeIndex,
    horizons: range | tuple[int, ...],
    include_target: bool = True,
) -> pd.DataFrame:
    """One row per (item_id, origin_date, horizon).

    `wide` is a date x item_id units matrix (see io.load_sales_wide).
    `calendar` is date-indexed with CALENDAR_COLS, covering every
    origin_date + max(horizons).
    """
    origin_dates = pd.DatetimeIndex(origin_dates)
    if len(origin_dates) == 0:
        raise ValueError("origin_dates must not be empty.")
    if not horizons:
        raise ValueError("horizons must not be empty.")
    missing_origins = origin_dates.difference(wide.index)
    if len(missing_origins):
        raise ValueError(f"origin date(s) not in the sales index: {list(missing_origins[:3])}.")
    if wide.isna().to_numpy().any():
        raise ValueError("wide has missing (date, SKU) cells; fix upstream data first.")
    missing_cal_cols = [c for c in CALENDAR_COLS if c not in calendar.columns]
    if missing_cal_cols:
        raise ValueError(f"calendar is missing column(s) {missing_cal_cols}.")

    max_h = max(horizons)
    target_dates = origin_dates + pd.Timedelta(days=max_h)
    missing_targets = target_dates.difference(calendar.index)
    if len(missing_targets):
        raise ValueError(
            f"calendar has no row for {len(missing_targets)} target date(s) "
            f"(e.g. {list(missing_targets[:3])}); an origin's horizon reaches past known dates."
        )

    lag_frames = {f"lag_{k}": wide.shift(k - 1) for k in LAGS}
    roll_frames = {f"rolling_mean_{w}": wide.rolling(w).mean() for w in ROLLING_WINDOWS}
    roll_frames["rolling_std_7"] = wide.rolling(7).std()
    roll_frames["rolling_zero_share_28"] = (wide == 0).rolling(28).mean()
    feature_frames = {**lag_frames, **roll_frames}

    base = None
    for name, frame in feature_frames.items():
        col = (
            frame.loc[origin_dates]
            .rename_axis(index="origin_date", columns="item_id")
            .stack()
            .rename(name)
        )
        base = col.to_frame() if base is None else base.join(col, how="outer")
    base = base.reset_index()
    # drops (item, origin) pairs without a full lag_28 lookback -- expected near the
    # start of the series, not an error.
    base = base.dropna(subset=list(feature_frames)).reset_index(drop=True)

    units_stacked = wide.rename_axis(index="date", columns="item_id").stack().rename("actual")

    pieces = []
    for h in horizons:
        piece = base.copy()
        piece["horizon"] = h
        h_target_dates = piece["origin_date"] + pd.Timedelta(days=h)

        cal = calendar.reindex(h_target_dates.to_numpy())
        piece["target_wday"] = cal["wday"].to_numpy()
        piece["target_month"] = cal["month"].to_numpy()
        piece["target_event_type"] = cal["event_type_1"].fillna("none").to_numpy()
        piece["target_snap"] = cal["snap"].to_numpy()

        if include_target:
            lookup = pd.MultiIndex.from_arrays([h_target_dates.to_numpy(), piece["item_id"].to_numpy()])
            piece["actual"] = units_stacked.reindex(lookup).to_numpy()

        pieces.append(piece)

    return pd.concat(pieces, ignore_index=True)
