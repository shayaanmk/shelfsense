"""LightGBM demand forecaster (build plan step 2b): lag/rolling/calendar
features, one global model across all SKUs with `horizon` as a feature --
a direct multi-horizon forecast (no recursive error accumulation).

Time-split into train/val/test blocks, chronologically ordered and
non-overlapping. The test block uses forecasting.baseline's exact origin
math (same STEP/N_ORIGINS/HORIZONS/SEASON), so lgbm_metrics.csv is directly
comparable to baseline_metrics.csv, SKU for SKU and horizon for horizon.

Modeling choices worth being able to defend (own these, per CLAUDE.md):
  - objective="tweedie": standard choice for sparse, non-negative demand
    counts (the M5 competition's own winning solutions used it); swap to
    "regression" (plain RMSE) or "poisson" in LGB_PARAMS to compare.
  - feature set: 4 weekly lags, 7/28-day rolling mean, 7-day rolling std,
    28-day zero-share (an intermittency signal, motivated by the low-volume
    SKUs baseline.py flagged as worst-WAPE), plus target-date calendar
    features and item_id as a categorical.
  - hyperparameters below are reasonable first-pass defaults, not tuned.

Run:  python -m forecasting.lgbm_model
Output: data/processed/lgbm_metrics.csv, data/processed/lgbm_model.txt
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting import baseline, features, io

VAL_N_ORIGINS = 8
GAP_DAYS = max(baseline.HORIZONS)  # keeps each block's forecast horizon from reaching into the next block

NUMERIC_FEATURES = [
    *[f"lag_{k}" for k in features.LAGS],
    *[f"rolling_mean_{w}" for w in features.ROLLING_WINDOWS],
    "rolling_std_7",
    "rolling_zero_share_28",
    "horizon",
    "target_snap",
]
CATEGORICAL_FEATURES = ["item_id", "target_wday", "target_month", "target_event_type"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50


def time_split(n_dates: int) -> tuple[list[int], list[int], list[int]]:
    """Chronologically ordered, non-overlapping origin blocks: train < val < test.

    A GAP_DAYS buffer separates each block so that no origin's forecast
    horizon reaches past its own block into the next one.
    """
    max_h = max(baseline.HORIZONS)

    test_idxs = baseline.origin_indices(n_dates, max_h, baseline.STEP, baseline.N_ORIGINS, baseline.SEASON)

    val_last_idx = min(test_idxs) - GAP_DAYS
    val_idxs = [val_last_idx - baseline.STEP * i for i in range(VAL_N_ORIGINS)]

    train_last_idx = min(val_idxs) - GAP_DAYS
    train_first_idx = features.MIN_HISTORY - 1
    if train_last_idx <= train_first_idx:
        raise ValueError(
            f"Not enough history to carve train/val/test blocks: only {n_dates} dates available."
        )
    train_idxs = list(range(train_first_idx, train_last_idx + 1))

    return train_idxs, val_idxs, test_idxs


def category_dtypes(item_ids: pd.Index, calendar: pd.DataFrame) -> dict[str, pd.CategoricalDtype]:
    """One shared category set per column, fit on the full dataset -- so train,
    val, test, and later inference (tools.get_forecast) all encode the same
    category to the same integer code. Fitting categories independently per
    call is a classic LightGBM pitfall: a category missing from one split
    gets a different code than in another, silently corrupting predictions.
    Callers must pass the same `calendar` (load_calendar's historical-only
    frame) used here at training time -- not a wider one -- so the category
    *set* stays identical; a wider calendar is fine for looking up feature
    *values* (see tools.get_forecast), just not for fitting these dtypes."""
    return {
        "item_id": pd.CategoricalDtype(sorted(item_ids)),
        "target_wday": pd.CategoricalDtype(sorted(calendar["wday"].unique())),
        "target_month": pd.CategoricalDtype(sorted(calendar["month"].unique())),
        "target_event_type": pd.CategoricalDtype(sorted(calendar["event_type_1"].fillna("none").unique())),
    }


def apply_category_dtypes(df: pd.DataFrame, dtypes: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    df = df.copy()
    for col, dtype in dtypes.items():
        df[col] = df[col].astype(dtype)
    return df


def load_calendar(long_df: pd.DataFrame) -> pd.DataFrame:
    """Calendar joined from sales_long.parquet -- covers only historical
    dates with a matching sales row. Used to fit category_dtypes at training
    time; tools.get_forecast needs io.load_future_calendar instead when it
    needs to look up values for dates past the last sales day."""
    return (
        long_df.drop_duplicates("date")
        .set_index("date")[features.CALENDAR_COLS]
        .sort_index()
    )


def main() -> None:
    long_df = io.load_sales_long()
    wide = io.load_sales_wide()
    calendar = load_calendar(long_df)
    dates = wide.index

    train_idxs, val_idxs, test_idxs = time_split(len(dates))
    print(
        f"Train: {len(train_idxs)} origins ({dates[min(train_idxs)].date()}..{dates[max(train_idxs)].date()})\n"
        f"Val:   {len(val_idxs)} origins ({dates[min(val_idxs)].date()}..{dates[max(val_idxs)].date()})\n"
        f"Test:  {len(test_idxs)} origins ({dates[min(test_idxs)].date()}..{dates[max(test_idxs)].date()})"
    )

    horizons = range(1, max(baseline.HORIZONS) + 1)
    dtypes = category_dtypes(wide.columns, calendar)
    train_df = apply_category_dtypes(features.build_feature_table(wide, calendar, dates[train_idxs], horizons), dtypes)
    val_df = apply_category_dtypes(features.build_feature_table(wide, calendar, dates[val_idxs], horizons), dtypes)
    test_df = apply_category_dtypes(features.build_feature_table(wide, calendar, dates[test_idxs], horizons), dtypes)

    train_set = lgb.Dataset(
        train_df[FEATURE_COLS], label=train_df["actual"], categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False
    )
    val_set = lgb.Dataset(
        val_df[FEATURE_COLS],
        label=val_df["actual"],
        categorical_feature=CATEGORICAL_FEATURES,
        reference=train_set,
        free_raw_data=False,
    )

    print(f"\nTraining on {len(train_df):,} rows, validating on {len(val_df):,} rows...")
    booster = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(period=100)],
    )

    test_df = test_df.copy()
    test_df["lgbm_pred"] = np.clip(
        booster.predict(test_df[FEATURE_COLS], num_iteration=booster.best_iteration), 0, None
    )

    results = test_df.rename(columns={"horizon": "day_offset"})[
        ["item_id", "origin_date", "day_offset", "actual", "lgbm_pred"]
    ]
    metrics = baseline.compute_metrics(results, horizons=baseline.HORIZONS, pred_cols=("lgbm_pred",))

    io.ensure_processed()
    out = io.PROCESSED / "lgbm_metrics.csv"
    metrics.to_csv(out, index=False)
    print(f"\nWrote {out}")

    model_path = io.PROCESSED / "lgbm_model.txt"
    booster.save_model(str(model_path))
    print(f"Wrote {model_path}")

    _print_comparison(metrics)
    _print_feature_importance(booster)


def _print_comparison(lgbm_metrics: pd.DataFrame) -> None:
    baseline_path = io.baseline_metrics_path()
    if not baseline_path.exists():
        print("\n(No baseline_metrics.csv found -- run `python -m forecasting.baseline` to compare.)")
        return
    base = pd.read_csv(baseline_path)
    sn = base[base["method"] == "seasonal_naive"]

    merged = lgbm_metrics.merge(sn, on=["item_id", "horizon"], suffixes=("_lgbm", "_sn"))

    print("\n=== LightGBM vs. seasonal_naive baseline (WAPE) ===")
    for horizon in baseline.HORIZONS:
        h = merged[merged["horizon"] == horizon]
        improvement = (h["wape_sn"] - h["wape_lgbm"]) / h["wape_sn"] * 100
        print(
            f"  {horizon}d: seasonal_naive median WAPE={h['wape_sn'].median():.1f}%  "
            f"lgbm median WAPE={h['wape_lgbm'].median():.1f}%  "
            f"median improvement={improvement.median():.1f}%  "
            f"({(improvement > 0).sum()}/{len(h)} SKUs improved)"
        )

    merged["wape_improvement_pct"] = (merged["wape_sn"] - merged["wape_lgbm"]) / merged["wape_sn"] * 100
    worse = merged[merged["wape_improvement_pct"] < 0].nsmallest(5, "wape_improvement_pct")
    if len(worse):
        print("\n  Worst regressions vs. seasonal_naive:")
        print(
            worse[["item_id", "horizon", "wape_sn", "wape_lgbm", "wape_improvement_pct"]].to_string(index=False)
        )


def _print_feature_importance(booster: lgb.Booster, top_n: int = 10) -> None:
    importance = pd.Series(
        booster.feature_importance(importance_type="gain"), index=booster.feature_name()
    ).sort_values(ascending=False)
    print(f"\n=== Top {top_n} features by gain ===")
    print(importance.head(top_n).round(1).to_string())


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
