---
name: testing-forecasting-pipeline
description: How to run and adversarially test the shelfsense CLI forecasting pipeline (forecasting.prepare_data / forecasting.baseline) without the real M5 Kaggle dataset.
---

# Testing the shelfsense forecasting pipeline

## Environment
- Use the repo venv: `.venv/bin/python` (blueprint's `pip install -e ".[dev]"` will fail — there is no
  `pyproject.toml`; install with `.venv/bin/pip install -r requirements.txt` instead).
- `pytest` gotcha: `.venv/bin/pytest tests -q` fails with `ModuleNotFoundError: No module named 'forecasting'`
  because the repo has no `conftest.py` / `pyproject.toml` `pythonpath`. Run `.venv/bin/python -m pytest tests -q`
  (adds CWD to `sys.path`), or add a root `conftest.py`.
- Both scripts resolve `data/raw` and `data/processed` **relative to CWD** — always run from a directory
  that contains those, e.g. `cd <repo> && .venv/bin/python -m forecasting.prepare_data`.

## Data
`data/raw` is gitignored and the Kaggle M5 CSVs are usually absent. Generate synthetic M5-style CSVs:
- `sales_train_evaluation.csv`: columns `id,item_id,dept_id,cat_id,store_id,state_id,d_1..d_N`
- `calendar.csv`: `d,date,wm_yr_wk,wday,month,year,event_name_1,event_type_1,snap_CA,snap_TX,snap_WI`
  (one row per `d_*`, contiguous dates)
- `sell_prices.csv`: `store_id,item_id,wm_yr_wk,sell_price` (existence-checked only)
Backtest defaults (`SEASON=7, STEP=7, N_ORIGINS=8, HORIZONS=(7,28)`) need **≥ 84 days**; use ~200 days
for a happy path. `CONFIG` in `forecasting/prepare_data.py` sets category/store/n_skus.

## Adversarial harness pattern
Never mutate the repo's own `data/raw`. For each failure case, create `/tmp/ss/<case>/` with a copy of
`forecasting/` + a mutated `data/raw/` + empty `data/processed/`, then run the module from that dir.
Config-driven cases (bad store, oversized `n_skus`) are best exercised by `sed`-editing `CONFIG` in the
per-case copy so the real `__main__` error wrapper (exit 1 + `ERROR: ...` on stderr) is exercised.
Regression baselining: `git worktree add /tmp/ss_main origin/main`, copy the same raw CSVs in, run both
scripts, and `diff` `data/processed/baseline_metrics.csv` — byte-identical means guards didn't change numbers.

## Known weak spot
A non-numeric **string** in a `d_*` column (e.g. `abc`) crashes with a pandas `TypeError` traceback at
`sales[day_cols].sum(axis=1)` *before* the `pd.to_numeric` guard runs, so it is not converted into the
clean `ERROR: ... non-numeric unit value(s) ...` line. Blank/empty cells do hit the guard. Test both.
