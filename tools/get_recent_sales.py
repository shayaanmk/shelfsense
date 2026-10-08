"""get_recent_sales(sku, days): real daily sales history for one of the
agent's tools -- reads data/processed/sales_long.parquet directly, no model
involved.
"""

from __future__ import annotations

from forecasting import io


def get_recent_sales(sku: str, days: int = 28) -> dict:
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}.")

    wide = io.load_sales_wide()
    if sku not in wide.columns:
        raise ValueError(f"Unknown SKU {sku!r}; {wide.shape[1]} SKUs are in scope, see data/processed/skus.txt.")
    if days > len(wide.index):
        raise ValueError(f"Only {len(wide.index)} days of history exist; requested {days}.")

    series = wide[sku]
    recent = series.iloc[-days:]

    trend_pct = None
    if len(series) >= 2 * days:
        prior = series.iloc[-2 * days : -days]
        if prior.mean() > 0:
            trend_pct = round(float((recent.mean() - prior.mean()) / prior.mean() * 100), 1)

    return {
        "sku": sku,
        "as_of_date": str(series.index[-1].date()),
        "days": days,
        "sales": [{"date": str(d.date()), "units": int(u)} for d, u in recent.items()],
        "summary": {
            "total_units": int(recent.sum()),
            "mean_daily_units": round(float(recent.mean()), 2),
            "trend_pct_vs_prior_period": trend_pct,
        },
    }
