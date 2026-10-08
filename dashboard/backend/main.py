"""FastAPI backend for the dashboard (build plan step 9) -- thin HTTP
wrappers around the real functions already built in tools/, agent/, and
commerce/. No business logic lives here; every endpoint just calls an
existing, already-tested function and translates its ValueError/
FileNotFoundError into an HTTP error instead of a raw 500.

This also doubles as the shape step 10's Azure Functions deployment will
wrap: copilot chat as the HTTP-triggered function, the rest as the API
surface a timer-triggered restock_agent run would otherwise bypass.

Run:  uvicorn dashboard.backend.main:app --reload --port 8000
(from the repo root, with the venv active)
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import restock_agent
from agent.copilot import ask as copilot_ask
from commerce import inventory, orders, pos
from forecasting import io
from tools.detect_anomalies import detect_anomalies
from tools.get_forecast import get_forecast
from tools.get_inventory_level import get_inventory_level
from tools.get_recent_sales import get_recent_sales

app = FastAPI(title="shelfsense dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _wrap(fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/skus")
def list_skus() -> dict:
    wide = io.load_sales_wide()
    return {"skus": sorted(wide.columns), "commerce_sku": pos.SKU}


@app.get("/api/forecast")
def forecast(sku: str, horizon: int = 7) -> dict:
    return _wrap(get_forecast, sku, horizon)


@app.get("/api/sales")
def sales(sku: str, days: int = 28) -> dict:
    return _wrap(get_recent_sales, sku, days)


@app.get("/api/inventory")
def inventory_level(sku: str) -> dict:
    return _wrap(get_inventory_level, sku)


@app.get("/api/anomalies")
def anomalies(sku: str, lookback_days: int = 28) -> dict:
    return _wrap(detect_anomalies, sku, lookback_days)


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] | None = None
    model: str | None = None


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    history = [m.model_dump() for m in req.history] if req.history else None
    kwargs = {"history": history}
    if req.model:
        kwargs["model"] = req.model
    result = _wrap(copilot_ask, req.message, **kwargs)
    # messages contains raw Anthropic content blocks (not JSON-serializable as-is
    # for tool_use blocks); the frontend only needs the trace + answer to render,
    # and resends its own plain-text history on the next turn.
    return {"answer": result["answer"], "trace": result["trace"]}


@app.get("/api/commerce/state")
def commerce_state() -> dict:
    state = _wrap(inventory.load_state)
    days = pos.trading_days()
    return {
        **state,
        "sim_start": str(days[0].date()),
        "sim_end": str(days[-1].date()),
        "finished": inventory.is_finished(),
    }


@app.get("/api/commerce/log")
def commerce_log() -> dict:
    log = inventory.load_log()
    log["date"] = log["date"].astype(str)
    return {"log": log.to_dict(orient="records")}


@app.get("/api/commerce/orders")
def commerce_orders() -> dict:
    df = orders.all_orders(pos.SKU)
    for col in ("order_date", "expected_arrival_date"):
        df[col] = df[col].astype(str)
    return {"orders": df.to_dict(orient="records")}


@app.get("/api/commerce/decisions")
def commerce_decisions() -> dict:
    df = restock_agent.load_decisions()
    return {"decisions": df.to_dict(orient="records")}


@app.post("/api/commerce/advance")
def commerce_advance() -> dict:
    if inventory.is_finished():
        raise HTTPException(status_code=400, detail="Simulation has reached the end of the replay window.")
    decision = restock_agent.run()
    day_result = inventory.advance_day()
    return {"decision": decision, "day_result": day_result, "state": inventory.load_state()}


@app.post("/api/commerce/reset")
def commerce_reset() -> dict:
    import os

    if os.path.exists(orders.LEDGER_PATH):
        os.remove(orders.LEDGER_PATH)
    if os.path.exists(restock_agent.DECISIONS_PATH):
        os.remove(restock_agent.DECISIONS_PATH)
    return {"state": inventory.reset()}
