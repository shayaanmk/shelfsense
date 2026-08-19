# shelfsense

Supply chain demand forecasting + an agent copilot that answers a planner's
natural-language questions grounded in real forecast/inventory/sales data.

See [CLAUDE.md](CLAUDE.md) for the full problem statement, architecture, and build plan.

## Status

**Step 1 — scope + data (in progress).** Repo scaffolding and the M5 subsetting
script are in place. Next: install Python, drop the dataset in, and run the prep script.

## Getting started

### 1. Install Python

Python is not yet installed on this machine (the `python` command currently
resolves to a Windows Store stub). Install Python 3.11+ from
[python.org](https://www.python.org/downloads/) or via `winget install Python.Python.3.12`,
then reopen your terminal.

### 2. Create a virtual environment and install deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Get the M5 dataset

Download **m5-forecasting-accuracy** from
[Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and
unzip these CSVs into `data/raw/`:

- `sales_train_evaluation.csv` (or `sales_train_validation.csv`)
- `calendar.csv`
- `sell_prices.csv`

`data/raw/` is git-ignored — the dataset never gets committed.

### 4. Subset and reshape

```powershell
python -m forecasting.prepare_data
```

Writes `data/processed/sales_long.parquet` (long-format sales for one
category/store, top-50 SKUs) and `data/processed/skus.txt`. Edit the `CONFIG`
block at the top of [forecasting/prepare_data.py](forecasting/prepare_data.py)
to re-scope (category / store / SKU count).

## Layout

```
data/          raw + processed M5 data (git-ignored)
forecasting/   feature engineering, baseline + LightGBM model, backtesting
tools/         get_forecast(), get_recent_sales(), get_inventory_level(), detect_anomalies()
agent/         hand-written plan-act-observe loop, calls tools/
eval/          synthetic anomaly injection, detection metrics, grounded-explanation judge
dashboard/     React frontend: forecast chart, anomaly flags, chat panel
deploy/        Azure Functions + App Service config
```
