"""Grounded-explanation check for agent.copilot (build plan step 6).

CLAUDE.md allows "rubric or LLM-judge." This uses a rubric: extract every
number the copilot's final answer states, and check it appears (within
tolerance) somewhere in that question's own tool-call trace. This is
deliberately the cheap, deterministic option -- no second LLM call to judge
the first one, so no added API cost and no judge-noise on top of the thing
being judged. It directly tests the most severe failure mode named in
CLAUDE.md's problem statement: a fabricated number that didn't come from any
tool.

What it does NOT catch (own this limitation, per CLAUDE.md): it can't tell
whether a cited number is attached to the right entity/date (e.g. citing a
real on_hand figure but mislabeling which day it's from), and it can't judge
open-ended reasoning ("this suggests supply risk") that has no number to
check. An LLM-judge would cover more of that at the cost of real API calls
and its own unreliability; swapping one in later is straightforward since
BENCHMARK_QUESTIONS and the scoring loop are separate from the rubric itself.

The benchmark set is intentionally small (see BENCHMARK_QUESTIONS) and runs
on Haiku by default -- this step makes real API calls, unlike anomaly_eval,
so cost is a real constraint, not just a preference.

Run:  python -m eval.grounding_eval
Output: data/processed/eval_grounding.csv
"""

from __future__ import annotations

import re

from agent.copilot import ask

MODEL = "claude-haiku-4-5-20251001"
NUMBER_TOLERANCE_ABS = 0.5
NUMBER_TOLERANCE_REL = 0.02

# Stripped before number extraction so neither gets misread as claimed numeric facts:
# an ISO date like "2016-05-28" otherwise splits into [2016.0, -5.0, -28.0], and a SKU
# id like "FOODS_1_018" otherwise yields [1.0, 18.0] -- both false "unsupported" flags.
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONTHS = (
    "Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    "Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
PROSE_DATE_RE = re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:,\s*\d{{4}})?\b")
SKU_RE = re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d[\d,]*\.?\d*")

# Covers each tool once, a multi-tool question, an out-of-scope SKU (tests honest
# refusal rather than fabrication), and a no-SKU question (tests that it asks for
# clarification per the system prompt rather than guessing which SKU).
BENCHMARK_QUESTIONS = [
    "What's the 7-day forecast for FOODS_1_018?",
    "What were the last 14 days of sales for FOODS_3_288?",
    "What's the current inventory level for FOODS_1_018?",
    "Have there been any anomalies for FOODS_1_018 in the last 90 days?",
    "Is FOODS_1_018 at risk of stocking out soon, and what's driving that?",
    "What's the forecast for SKU FOODS_9_999?",
    "How is demand trending right now?",
]


def _numbers_in(text: str) -> list[float]:
    text = DATE_RE.sub(" ", text)
    text = PROSE_DATE_RE.sub(" ", text)
    text = SKU_RE.sub(" ", text)
    return [float(m.replace(",", "")) for m in NUMBER_RE.findall(text)]


def _flatten_numbers(value) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _flatten_numbers(v)]
    if isinstance(value, (list, tuple)):
        return [n for v in value for n in _flatten_numbers(v)]
    if isinstance(value, str):
        return _numbers_in(value)
    return []


def _is_grounded(claimed: float, evidence_numbers: list[float]) -> bool:
    tolerance = max(NUMBER_TOLERANCE_ABS, abs(claimed) * NUMBER_TOLERANCE_REL)
    return any(abs(claimed - n) <= tolerance for n in evidence_numbers)


def score_answer(answer: str, trace: list[dict]) -> dict:
    evidence_numbers = [n for call in trace for n in _flatten_numbers(call["output"])]
    claimed_numbers = _numbers_in(answer)

    # small integers (1-12) are cheap to state honestly without a tool (dates, counts,
    # "7 days", list indices) and flag constantly as false "unsupported" hits; exclude them.
    checkable = [n for n in claimed_numbers if abs(n) > 12 or n != int(n)]
    unsupported = [n for n in checkable if not _is_grounded(n, evidence_numbers)]

    return {
        "n_claimed_numbers": len(checkable),
        "n_unsupported": len(unsupported),
        "unsupported_numbers": unsupported,
        "fully_grounded": len(unsupported) == 0,
    }


def run_eval(questions: list[str] = BENCHMARK_QUESTIONS, model: str = MODEL) -> list[dict]:
    rows = []
    for question in questions:
        result = ask(question, model=model)
        score = score_answer(result["answer"], result["trace"])
        rows.append(
            {
                "question": question,
                "n_tool_calls": len(result["trace"]),
                "tools_used": [c["tool"] for c in result["trace"]],
                "answer": result["answer"],
                **score,
            }
        )
    return rows


def main() -> None:
    print(f"Running {len(BENCHMARK_QUESTIONS)} benchmark questions on {MODEL}...")
    rows = run_eval()

    import pandas as pd

    from forecasting import io

    df = pd.DataFrame(rows)
    io.ensure_processed()
    out = io.PROCESSED / "eval_grounding.csv"
    df.drop(columns=["answer"]).to_csv(out, index=False)
    print(f"Wrote {out}\n")

    n_grounded = df["fully_grounded"].sum()
    print(f"=== {n_grounded}/{len(df)} answers fully grounded ===\n")
    for row in rows:
        status = "OK" if row["fully_grounded"] else "UNSUPPORTED NUMBERS"
        print(f"[{status}] {row['question']}")
        print(f"  tools: {row['tools_used']}")
        if not row["fully_grounded"]:
            print(f"  unsupported: {row['unsupported_numbers']}")
        print(f"  answer: {row['answer'][:200]}")
        print()


if __name__ == "__main__":
    main()
