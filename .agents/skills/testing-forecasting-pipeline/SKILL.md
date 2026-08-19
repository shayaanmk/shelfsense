---
name: testing-forecasting-pipeline
description: How to run and verify the ShelfSense forecasting CLI pipeline (forecasting.prepare_data / forecasting.baseline) end-to-end without the real Kaggle M5 dataset, including a byte-for-byte refactor-equivalence method.
---

# Testing the ShelfSense forecasting pipeline

There is no UI. Everything is `python3 -m forecasting.<module>` run **from the repo root**
(paths in `forecasting/io.py` are relative: `data/raw`, `data/processed`).

## Environment
- System `python3` (3.10) already has pandas, numpy and pyarrow importable — no venv needed.
  The repo blueprint's `python3 -m venv .venv` + `pip install -e ".[dev]"` may fail because the
  repo has no `pyproject.toml`; if it does, fall back to system python3 or
  `pip install -r requirements.txt` (note `lightgbm`/`anthropic` are not needed for
  prepare_data/baseline).
- Real Kaggle M5 data is not on the machine and there are no Kaggle credentials. Fabricate data.

## Fabricating an M5-shaped raw dataset
Write into `data/raw/` (git-ignored):
- `sales_train_evaluation.csv`: columns `id,item_id,dept_id,cat_id,store_id,state_id,d_1..d_N`.
  Include rows matching `CONFIG` in `forecasting/prepare_data.py` (`cat_id=FOODS`, `store_id=CA_1`)
  **plus decoy rows** with a different cat/store so you can assert filtering works.
- `calendar.csv`: `d,date,wm_yr_wk,wday,month,year,event_name_1,event_type_1,snap_CA`
  (snap column must match `snap_<first 2 chars of store>`).
- `sell_prices.csv`: existence-only check — a 1-row stub is enough.
- Size: `forecasting/baseline.py` needs `len(dates) - 1 - 28 - 7*7 >= 7`, i.e. **>= ~85 days**;
  140 days x ~6 SKUs is comfortable and fast.
- Use a seeded RNG (`np.random.default_rng(1234)`) so the fixture is reproducible across runs
  and across worktrees.

## Expected artifacts / assertions
- `prepare_data` prints "Kept N SKUs" and writes `data/processed/sales_long.parquet`
  (rows = kept SKUs x days) and `skus.txt` (one item_id per line, decoys absent).
  Parquet columns: id,item_id,dept_id,cat_id,store_id,state_id,d,units,date(datetime64),
  wm_yr_wk,wday,month,year,event_name_1,event_type_1,**snap** (renamed from snap_XX).
- `baseline` writes `data/processed/baseline_metrics.csv` with
  `SKUs x 2 methods x 2 horizons` rows, columns item_id,method,horizon,rmse,mape,wape,n_obs,
  n_zero_actual; `n_obs = n_origins * horizon` (default 8*7=56 and 8*28=224).

## Refactor-equivalence method (best assertion for pure refactors)
```
git worktree add /tmp/ss-main main
cp -r data/raw /tmp/ss-main/data/
(cd /tmp/ss-main && python3 -m forecasting.prepare_data && python3 -m forecasting.baseline)
sha256sum {,/tmp/ss-main/}data/processed/baseline_metrics.csv   # must match
```
Both parquet and CSV outputs are deterministic, so hashes match byte-for-byte if behavior is
unchanged. Clean up with `git worktree remove /tmp/ss-main --force`.

## Error paths worth checking
- No `data/processed/sales_long.parquet` → `FileNotFoundError: ... run
  `python -m forecasting.prepare_data` first.`
- Empty `data/raw/` → `None of ['sales_train_evaluation.csv', 'sales_train_validation.csv']
  found in data/raw/. Download the M5 dataset ...`
- Missing `calendar.csv` / `sell_prices.csv` → `Expected data/raw/<name>; Download the M5 ...`

## Cleanup
`data/raw/*` and `data/processed/*` are git-ignored but should still be emptied afterwards
(keep `data/processed/.gitkeep`). Never commit fixtures.

## Devin Secrets Needed
None. (Kaggle credentials would only be needed to test against the real M5 dataset.)
