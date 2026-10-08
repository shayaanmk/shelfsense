# shelfsense

Supply chain demand forecasting with two agents on top: an **interactive
copilot** that answers a planner's natural-language questions grounded in
real forecast/inventory/sales data, and an **autonomous restocking agent**
that decides order quantity and supplier on a schedule, with no human in
the loop before it fires.

See [CLAUDE.md](CLAUDE.md) for the full problem statement, architecture, and build plan.

## Status

Steps 1–9 of the build plan are done: data scoping, forecasting baseline +
LightGBM model, the eval harness (anomaly-injection precision/recall +
grounded-explanation check), the copilot's tool interface and agent loop,
the commerce simulation, the restocking agent, and this dashboard. Steps 10
(Azure deployment) and 11 (write-up) are next.

## Getting started

### 1. Install Python

Install Python 3.11+ from [python.org](https://www.python.org/downloads/)
or via `winget install Python.Python.3.12`, then reopen your terminal.

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

### 4. Build the data + model pipeline

```powershell
python -m forecasting.prepare_data      # subset M5 to one category/store, ~50 SKUs
python -m forecasting.baseline          # naive + seasonal-naive baselines, backtested
python -m forecasting.lgbm_model        # LightGBM model, trained + backtested
python -m tools.inventory_sim           # retrospective (s,S) inventory sim, backs detect_anomalies' stockout signal
```

Each writes its output to `data/processed/`. Re-run in order after changing
`CONFIG` in [forecasting/prepare_data.py](forecasting/prepare_data.py) (category / store / SKU count).

### 5. Set up the copilot

The copilot calls the Claude API (`agent/copilot.py`). Add a `.env` file in
the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Then try it:

```powershell
python -m agent.copilot "Is FOODS_1_018 at risk of stocking out soon?"
```

### 6. Run the evaluation harness

```powershell
python -m eval.anomaly_eval       # free -- synthetic anomaly injection, precision/recall
python -m eval.grounding_eval     # costs real API calls (small benchmark, Haiku by default)
```

### 7. Run the commerce simulation + restocking agent

```powershell
python -m tools.inventory_sim                          # (if not already run above)
python -c "from commerce import inventory; inventory.reset()"
python -m agent.restock_agent                           # decide once for the current simulated day
python -m eval.restock_eval                              # restock_agent vs. naive baseline, simulated profit
```

### 8. Run the dashboard

Backend (FastAPI):

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn dashboard.backend.main:app --reload --port 8000
```

Frontend (React + Vite), in a second terminal:

```powershell
cd dashboard\frontend
npm install
npm run dev
```

Open the URL Vite prints (typically http://localhost:5173).

## Layout

```
data/            raw + processed M5 data (git-ignored)
forecasting/     feature engineering, baseline + LightGBM model, backtesting
tools/           get_forecast(), get_recent_sales(), get_inventory_level(),
                 detect_anomalies() -- real functions over the data, not mocks
commerce/        pos.py, inventory.py, suppliers.py, orders.py -- the single-SKU
                 commerce simulation the restocking agent operates on
agent/           copilot.py (interactive, request/response) and
                 restock_agent.py (autonomous, schedule-triggered)
eval/            synthetic anomaly injection + precision/recall, grounded-explanation
                 check, restock_agent-vs-naive-baseline profit comparison
dashboard/       backend/ (FastAPI) + frontend/ (React + Vite)
deploy/          Azure Functions + App Service config
```
