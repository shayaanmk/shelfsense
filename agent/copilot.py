"""Hand-written plan-act-observe agent loop (build plan step 5) -- no LangChain.

A planner asks a natural-language question; the loop lets Claude decide which
of the four tools/ functions to call (get_forecast, get_recent_sales,
get_inventory_level, detect_anomalies), executes them for real, feeds the
results back, and repeats until Claude has enough to answer -- or gives up
after MAX_ITERATIONS and says so, rather than guessing.

Control-flow choices worth being able to defend (own these, per CLAUDE.md):
  - Stateless per question, with an optional `history` param for a future
    multi-turn chat panel (step 9) to thread through -- the loop itself
    doesn't need to remember past questions to do its job.
  - MAX_ITERATIONS=6 is a runaway-loop safety cap, not an expected depth --
    a single question rarely needs more than 1-2 tool calls across these
    four tools.
  - A tool call that raises (bad SKU, bad horizon) is caught and fed back as
    an `is_error` tool_result, not a crash -- Claude sees the real error
    message and can recover (ask the planner to clarify) or report it.
  - ask() returns the full {answer, trace, messages} rather than just the
    text, so callers (eval harness step 6, dashboard step 9) can inspect
    exactly which tool calls grounded the answer, not just trust the prose.
  - MODEL defaults to Sonnet for answer quality; step 6's eval harness will
    make many more calls per run, so swap to Haiku there if cost matters --
    one constant to change, not a design fork.

Run:  python -m agent.copilot "why did SKU FOODS_1_018 stock out recently?"
Requires ANTHROPIC_API_KEY (in the environment or a .env file).
"""

from __future__ import annotations

import json
import sys

import anthropic
from dotenv import load_dotenv

from forecasting import io
from forecasting.prepare_data import CONFIG as SCOPE_CONFIG
from tools.detect_anomalies import detect_anomalies
from tools.get_forecast import get_forecast
from tools.get_inventory_level import get_inventory_level
from tools.get_recent_sales import get_recent_sales

load_dotenv()

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
MAX_ITERATIONS = 6

TOOL_REGISTRY = {
    "get_forecast": get_forecast,
    "get_recent_sales": get_recent_sales,
    "get_inventory_level": get_inventory_level,
    "detect_anomalies": detect_anomalies,
}

TOOLS = [
    {
        "name": "get_forecast",
        "description": (
            "Demand forecast for one SKU over a horizon of days, from the trained "
            "LightGBM model (or the seasonal-naive baseline if the model isn't "
            "trained yet). Predicted daily units starting the day after the most "
            "recent known sales date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "SKU item_id, e.g. 'FOODS_1_018'."},
                "horizon": {"type": "integer", "description": "Days ahead to forecast, 1-28."},
            },
            "required": ["sku", "horizon"],
        },
    },
    {
        "name": "get_recent_sales",
        "description": "Real daily sales history for one SKU over the last N days, with a trend summary vs. the prior period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "SKU item_id, e.g. 'FOODS_1_018'."},
                "days": {"type": "integer", "description": "Number of most-recent days to return. Default 28."},
            },
            "required": ["sku"],
        },
    },
    {
        "name": "get_inventory_level",
        "description": (
            "Simulated on-hand inventory for one SKU (M5 has no real inventory data, so this is "
            "a reorder-point policy simulation driven by real sales -- see tools/inventory_sim.py). "
            "Returns on-hand units, days of supply, reorder point, target stock level, whether it "
            "stocked out that day, and the last restock date. Defaults to the most recent date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "SKU item_id, e.g. 'FOODS_1_018'."},
                "as_of": {"type": "string", "description": "Optional ISO date (YYYY-MM-DD) for a historical lookup."},
            },
            "required": ["sku"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": (
            "Flags anomalies for one SKU in the last N days: demand_spike/demand_drop (actual sales "
            "vs. the same weekday one week prior, by trailing z-score) and stockout (from the "
            "inventory simulation). Each anomaly includes its evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "SKU item_id, e.g. 'FOODS_1_018'."},
                "lookback_days": {"type": "integer", "description": "Days to scan for anomalies. Default 28."},
            },
            "required": ["sku"],
        },
    },
]


def _system_prompt() -> str:
    n_skus = len(io.load_sales_wide().columns)
    return (
        f"You are a supply chain planning copilot for one retail store "
        f"(category={SCOPE_CONFIG['category']}, store={SCOPE_CONFIG['store']}, {n_skus} SKUs in scope).\n\n"
        "A human planner will ask about demand forecasts, recent sales, inventory levels, or "
        "anomalies for a specific SKU (item_id, e.g. \"FOODS_1_018\"). You have four tools, each a "
        "real function over real data -- never state a number that didn't come from a tool result.\n\n"
        "Ground every claim in your final answer in the specific tool output that supports it "
        "(e.g. \"on 2016-04-29 sales dropped to 1 unit vs. 25 the prior week -- flagged as a "
        "demand_drop anomaly, z=-2.73\"). If the tools don't have enough information to answer, say "
        "so explicitly rather than speculating.\n\n"
        "If the question doesn't name a SKU, ask which one before calling any tool -- every tool "
        "requires one."
    )


def _run_tool(name: str, tool_input: dict) -> tuple[dict, bool]:
    """Executes one real tool call. Returns (result_dict, is_error) -- an error is
    reported back to the model as data, not raised, so a bad SKU or horizon becomes
    something the agent can react to instead of a crash."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name!r}."}, True
    try:
        return fn(**tool_input), False
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}, True


def ask(
    question: str,
    history: list[dict] | None = None,
    client: anthropic.Anthropic | None = None,
    model: str = MODEL,
) -> dict:
    """Answers one planner question, calling tools as needed.

    Returns {"answer": str, "trace": [...], "messages": [...]}. `trace` is a
    flat, log-friendly list of every tool call made (name, input, output,
    whether it errored); `messages` is the full Anthropic message history,
    useful for a caller that wants to continue the conversation.

    `model` defaults to MODEL (Sonnet) but is overridable per call -- e.g. a
    cheap smoke test or eval/'s high-volume runs can pass Haiku without
    changing the production default.
    """
    client = client or anthropic.Anthropic()
    messages = list(history) if history else []
    messages.append({"role": "user", "content": question})

    trace = []
    for _ in range(MAX_ITERATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=_system_prompt(),
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            answer = "".join(block.text for block in response.content if block.type == "text")
            return {"answer": answer, "trace": trace, "messages": messages}

        tool_results = []
        for block in tool_uses:
            result, is_error = _run_tool(block.name, block.input)
            trace.append({"tool": block.name, "input": block.input, "output": result, "is_error": is_error})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": (
            "I wasn't able to gather enough grounded information to answer that within "
            f"{MAX_ITERATIONS} tool-call rounds. Please try a more specific question."
        ),
        "trace": trace,
        "messages": messages,
    }


def _print_trace(trace: list[dict]) -> None:
    for i, call in enumerate(trace, 1):
        status = "ERROR" if call["is_error"] else "ok"
        print(f"  [{i}] {call['tool']}({call['input']}) -> {status}")
        print(f"      {json.dumps(call['output'])[:200]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m agent.copilot \"<question>\"", file=sys.stderr)
        raise SystemExit(1)

    result = ask(sys.argv[1])
    print("=== Tool-call trace ===")
    _print_trace(result["trace"])
    print("\n=== Answer ===")
    print(result["answer"])
