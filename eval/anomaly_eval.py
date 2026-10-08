"""Synthetic anomaly injection + detection precision/recall (build plan step 6).

Validates tools.detect_anomalies against KNOWN ground truth, which the real
M5 history doesn't give us -- we don't actually know which historical days
are "true" anomalies, only which ones the detector flags. Two injection
types, per CLAUDE.md step 6:

  - demand_spike / demand_drop: scale one real day's units by a large
    multiplier (or down to near zero), on a (SKU, date) chosen to already
    carry meaningful volume so the injected perturbation is a real jump,
    not noise around zero.
  - supply_delay: extend the lead time of whatever reorder
    tools.inventory_sim would have placed around a chosen date (see
    simulate_item's delay_orders_on param) -- a shipment delay as a real
    mechanism, not a flag painted onto the data. Only scenarios where the
    delay actually causes a NEW stockout (vs. the undelayed baseline) count;
    plenty of delays get absorbed by safety stock and aren't a usable test.

Injection dates are drawn from the test block in forecasting.lgbm_model's
time split -- genuinely held out from every model fit in this repo, demand
or inventory.

Precision/recall definition (own this, per CLAUDE.md): scoring flagged dates
against ONLY the injected date would wrongly count a detector's real,
pre-existing flags (unrelated to anything we injected) as false positives.
So for each scenario we run detect_anomalies on the series BEFORE injection
(baseline) and AFTER (injected), and count:
  - true positive:  the injected/ground-truth date is flagged after injection
  - false negative: it is NOT flagged after injection
  - false positive: a date other than the injected one is flagged after
    injection but was NOT flagged before it -- i.e. the injection created a
    spurious flag elsewhere, not a pre-existing one
A demand point-edit also perturbs the week-over-week residual at date+7
(since that day's "prior week" value is now the injected one) -- a
mechanical side effect of the detector's own design, not an independent
false positive, so dates within FP_EXCLUSION_RADIUS_DAYS of the injected
date are excluded from the false-positive count (the injected date itself
is still scored as TP/FN as usual).

Run:  python -m eval.anomaly_eval
Output: data/processed/eval_anomaly_detection.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting import baseline, io, lgbm_model
from tools import inventory_sim
from tools.detect_anomalies import Z_THRESHOLD, detect_anomalies

RNG_SEED = 0
N_SCENARIOS_PER_TYPE = 15

SPIKE_MULTIPLIER = 5.0
SPIKE_MIN_ABS = 10.0
DROP_MULTIPLIER = 0.1
SUPPLY_DELAY_EXTRA_DAYS = 7
FP_EXCLUSION_RADIUS_DAYS = 7


def _held_out_window(wide: pd.DataFrame) -> list[int]:
    """Day-indices covered by lgbm_model's test block -- never used to fit
    anything (the forecast model, the z-score threshold) in this repo."""
    _, _, test_idxs = lgbm_model.time_split(len(wide.index))
    start = min(test_idxs)
    end = max(test_idxs) + max(baseline.HORIZONS)
    return list(range(start, end + 1))


def _demand_candidates(wide: pd.DataFrame, window_idxs: list[int], rng: np.random.Generator, n: int) -> list[dict]:
    medians = wide.median()
    skus = list(wide.columns)
    seen = set()
    candidates = []
    attempts = 0
    while len(candidates) < n and attempts < n * 50:
        attempts += 1
        sku = rng.choice(skus)
        idx = int(rng.choice(window_idxs))
        if (sku, idx) in seen:
            continue
        original = float(wide.iloc[idx][sku])
        if original >= max(float(medians[sku]), 1.0):
            seen.add((sku, idx))
            candidates.append({"sku": sku, "idx": idx, "date": wide.index[idx], "original": original})
    return candidates


def _inject_demand_spike(wide: pd.DataFrame, sku: str, date: pd.Timestamp, original: float) -> pd.DataFrame:
    modified = wide.astype(float).copy()
    modified.loc[date, sku] = max(original * SPIKE_MULTIPLIER, original + SPIKE_MIN_ABS)
    return modified


def _inject_demand_drop(wide: pd.DataFrame, sku: str, date: pd.Timestamp, original: float) -> pd.DataFrame:
    modified = wide.astype(float).copy()
    modified.loc[date, sku] = original * DROP_MULTIPLIER
    return modified


def _supply_delay_scenarios(wide: pd.DataFrame, window_idxs: list[int], rng: np.random.Generator, n: int) -> list[dict]:
    dates = wide.index
    window_set = set(window_idxs)
    skus = list(wide.columns)
    rng.shuffle(skus)

    scenarios = []
    for sku in skus:
        if len(scenarios) >= n:
            break
        demand = wide[sku].to_numpy(dtype=float)
        s, target = inventory_sim.reorder_policy(demand)
        base_sim = inventory_sim.simulate_item(dates, demand, s, target)
        base_stockouts = set(base_sim.loc[base_sim["stockout"], "date"])

        order_idxs = [t for t in range(len(demand)) if base_sim["order_placed_qty"].iloc[t] > 0 and t in window_set]
        rng.shuffle(order_idxs)
        for t in order_idxs:
            if len(scenarios) >= n:
                break
            delayed_sim = inventory_sim.simulate_item(
                dates, demand, s, target, delay_orders_on={t}, extra_lead_days=SUPPLY_DELAY_EXTRA_DAYS
            )
            delayed_stockouts = set(delayed_sim.loc[delayed_sim["stockout"], "date"])
            new_stockouts = sorted(d for d in (delayed_stockouts - base_stockouts) if d > dates[t])
            if not new_stockouts:
                continue  # the delay was absorbed by safety stock; not a usable test case
            ground_truth_date = new_stockouts[0]
            delayed_inv = delayed_sim.copy()
            delayed_inv.insert(0, "item_id", sku)
            delayed_inv["reorder_point"] = s
            delayed_inv["target_level"] = target
            base_inv = base_sim.copy()
            base_inv.insert(0, "item_id", sku)
            base_inv["reorder_point"] = s
            base_inv["target_level"] = target
            scenarios.append(
                {
                    "sku": sku,
                    "ground_truth_date": ground_truth_date,
                    "order_date": dates[t],
                    "base_inventory": base_inv,
                    "injected_inventory": delayed_inv,
                }
            )
    return scenarios


def _lookback_for(wide: pd.DataFrame, idx: int) -> int:
    return len(wide.index) - idx


def _flagged_dates(result: dict) -> set[pd.Timestamp]:
    return {pd.Timestamp(a["date"]) for a in result["anomalies"]}


def _score_scenario(
    scenario_type: str,
    sku: str,
    ground_truth_date: pd.Timestamp,
    base_result: dict,
    injected_result: dict,
) -> dict:
    base_flagged = _flagged_dates(base_result)
    injected_flagged = _flagged_dates(injected_result)

    true_positive = ground_truth_date in injected_flagged
    exclusion = {ground_truth_date + pd.Timedelta(days=d) for d in range(-FP_EXCLUSION_RADIUS_DAYS, FP_EXCLUSION_RADIUS_DAYS + 1)}
    false_positive_dates = (injected_flagged - base_flagged) - exclusion

    return {
        "scenario_type": scenario_type,
        "sku": sku,
        "ground_truth_date": str(ground_truth_date.date()),
        "true_positive": true_positive,
        "false_negative": not true_positive,
        "n_false_positives": len(false_positive_dates),
    }


def run_eval(
    seed: int = RNG_SEED, n_per_type: int = N_SCENARIOS_PER_TYPE, z_threshold: float = Z_THRESHOLD
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wide = io.load_sales_wide()
    window_idxs = _held_out_window(wide)

    rows = []

    for scenario_type, inject_fn in (("demand_spike", _inject_demand_spike), ("demand_drop", _inject_demand_drop)):
        for cand in _demand_candidates(wide, window_idxs, rng, n_per_type):
            injected_wide = inject_fn(wide, cand["sku"], cand["date"], cand["original"])
            lookback = _lookback_for(wide, cand["idx"])
            base_result = detect_anomalies(cand["sku"], lookback, wide=wide, z_threshold=z_threshold)
            injected_result = detect_anomalies(cand["sku"], lookback, wide=injected_wide, z_threshold=z_threshold)
            rows.append(_score_scenario(scenario_type, cand["sku"], cand["date"], base_result, injected_result))

    for scenario in _supply_delay_scenarios(wide, window_idxs, rng, n_per_type):
        idx = wide.index.get_loc(scenario["ground_truth_date"])
        lookback = _lookback_for(wide, idx)
        base_result = detect_anomalies(
            scenario["sku"], lookback, wide=wide, inventory=scenario["base_inventory"], z_threshold=z_threshold
        )
        injected_result = detect_anomalies(
            scenario["sku"], lookback, wide=wide, inventory=scenario["injected_inventory"], z_threshold=z_threshold
        )
        rows.append(
            _score_scenario("supply_delay", scenario["sku"], scenario["ground_truth_date"], base_result, injected_result)
        )

    return pd.DataFrame(rows)


def sweep_thresholds(thresholds: list[float], seed: int = RNG_SEED, n_per_type: int = N_SCENARIOS_PER_TYPE) -> pd.DataFrame:
    """Recall/false-positive tradeoff across candidate Z_THRESHOLD values, on
    the SAME injected scenarios each time (same seed) -- isolates the effect
    of the threshold from scenario-sampling noise. Own the final pick, per
    CLAUDE.md; this just makes the tradeoff visible instead of guessed."""
    rows = []
    for z in thresholds:
        results = run_eval(seed=seed, n_per_type=n_per_type, z_threshold=z)
        summary = summarize(results)
        summary["z_threshold"] = z
        rows.append(summary.reset_index())
    return pd.concat(rows, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("scenario_type").agg(
        n_scenarios=("sku", "count"),
        recall=("true_positive", "mean"),
        total_false_positives=("n_false_positives", "sum"),
    )
    # precision here is scenario-level: of all flags the detector raised attributable to
    # the injection (TPs + attributable FPs), what fraction were the actual injected event.
    tp = results.groupby("scenario_type")["true_positive"].sum()
    fp = results.groupby("scenario_type")["n_false_positives"].sum()
    summary["precision"] = (tp / (tp + fp)).replace([np.inf, -np.inf], np.nan)
    return summary.round(3)


def main() -> None:
    print(f"Z_THRESHOLD currently {Z_THRESHOLD} -- injecting {N_SCENARIOS_PER_TYPE} scenarios per type...")
    results = run_eval()

    io.ensure_processed()
    out = io.PROCESSED / "eval_anomaly_detection.csv"
    results.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(results)} scenarios)\n")

    print("=== Precision / recall by scenario type ===")
    print(summarize(results))

    misses = results[~results["true_positive"]]
    if len(misses):
        print(f"\n=== Missed ({len(misses)}) ===")
        print(misses[["scenario_type", "sku", "ground_truth_date"]].to_string(index=False))

    print("\n=== Z_THRESHOLD sweep (same scenarios, different threshold) ===")
    sweep = sweep_thresholds([1.5, 2.0, 2.5, 3.0])
    sweep_out = io.PROCESSED / "eval_threshold_sweep.csv"
    sweep.to_csv(sweep_out, index=False)
    print(sweep.to_string(index=False))
    print(f"\nWrote {sweep_out}")


if __name__ == "__main__":
    main()
