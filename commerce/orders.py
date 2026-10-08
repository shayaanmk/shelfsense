"""PurchaseOrder model + ledger (build plan step 7, architecture's commerce/
orders.py). Scaffolding only: step 8's restocking agent decides WHAT to
order; this module just records and tracks orders once placed, and hands
back whatever has arrived by a given date.

Ledger persisted to data/commerce/po_ledger.csv -- a flat, append-friendly
table (not JSON) so a dashboard (step 9) can read it directly as a PO
history table without any reshaping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from forecasting import io

LEDGER_PATH = io.PROCESSED.parent / "commerce" / "po_ledger.csv"

LEDGER_COLUMNS = [
    "po_id",
    "sku",
    "supplier_id",
    "quantity",
    "unit_price",
    "total_cost",
    "order_date",
    "expected_arrival_date",
    "status",
    "rationale",
]


@dataclass
class PurchaseOrder:
    po_id: str
    sku: str
    supplier_id: str
    quantity: int
    unit_price: float
    order_date: str
    expected_arrival_date: str
    status: str  # "pending" | "received" | "cancelled"
    rationale: str | None = None

    @property
    def total_cost(self) -> float:
        return round(self.quantity * self.unit_price, 2)

    def to_row(self) -> dict:
        row = asdict(self)
        row["total_cost"] = self.total_cost
        return row


def _ensure_ledger() -> pd.DataFrame:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER_PATH.exists():
        pd.DataFrame(columns=LEDGER_COLUMNS).to_csv(LEDGER_PATH, index=False)
    return pd.read_csv(LEDGER_PATH, dtype={"po_id": str})


def load_ledger() -> pd.DataFrame:
    df = _ensure_ledger()
    for col in ("order_date", "expected_arrival_date"):
        df[col] = pd.to_datetime(df[col])
    return df


def _save_ledger(df: pd.DataFrame) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEDGER_PATH, index=False)


def place_order(
    sku: str,
    supplier_id: str,
    quantity: int,
    unit_price: float,
    order_date: str,
    lead_time_days: int,
    rationale: str | None = None,
    status: str = "pending",
) -> PurchaseOrder:
    """`status="pending_approval"` (used by agent.restock_agent's large-order
    guard) records a decision WITHOUT starting the lead-time clock -- it sits
    outside receive_due_orders' "pending" scan, so it never silently arrives;
    a human approving it is a step this project doesn't build a UI for yet,
    but the guard's real job -- not auto-executing -- is enforced regardless."""
    if quantity <= 0:
        raise ValueError(f"quantity must be > 0, got {quantity}.")
    if unit_price <= 0:
        raise ValueError(f"unit_price must be > 0, got {unit_price}.")
    if status not in ("pending", "pending_approval"):
        raise ValueError(f"status must be 'pending' or 'pending_approval', got {status!r}.")

    ledger = _ensure_ledger()
    po_id = f"PO-{len(ledger) + 1:04d}"
    expected_arrival = (pd.Timestamp(order_date) + pd.Timedelta(days=lead_time_days)).date()

    po = PurchaseOrder(
        po_id=po_id,
        sku=sku,
        supplier_id=supplier_id,
        quantity=int(quantity),
        unit_price=float(unit_price),
        order_date=str(pd.Timestamp(order_date).date()),
        expected_arrival_date=str(expected_arrival),
        status=status,
        rationale=rationale,
    )

    ledger = pd.concat([ledger, pd.DataFrame([po.to_row()])], ignore_index=True)
    _save_ledger(ledger)
    return po


def receive_due_orders(as_of_date: str) -> list[PurchaseOrder]:
    """Marks every still-pending order whose expected_arrival_date <=
    as_of_date as received, persists the change, and returns those orders
    -- commerce.inventory calls this each simulated day to know how much
    stock just arrived."""
    ledger = load_ledger()
    as_of = pd.Timestamp(as_of_date)
    due_mask = (ledger["status"] == "pending") & (ledger["expected_arrival_date"] <= as_of)
    due = ledger[due_mask]
    if due.empty:
        return []

    ledger.loc[due_mask, "status"] = "received"
    _save_ledger(ledger)

    return [
        PurchaseOrder(
            po_id=row.po_id,
            sku=row.sku,
            supplier_id=row.supplier_id,
            quantity=int(row.quantity),
            unit_price=float(row.unit_price),
            order_date=str(row.order_date.date()),
            expected_arrival_date=str(row.expected_arrival_date.date()),
            status="received",
            rationale=row.rationale if isinstance(row.rationale, str) else None,
        )
        for row in due.itertuples()
    ]


def pending_orders(sku: str | None = None) -> pd.DataFrame:
    ledger = load_ledger()
    pending = ledger[ledger["status"] == "pending"]
    return pending[pending["sku"] == sku] if sku else pending


def all_orders(sku: str | None = None) -> pd.DataFrame:
    ledger = load_ledger()
    return ledger[ledger["sku"] == sku] if sku else ledger


if __name__ == "__main__":
    print(f"Ledger: {LEDGER_PATH} ({len(load_ledger())} orders)")
    print(all_orders().to_string(index=False) if len(load_ledger()) else "(empty)")
