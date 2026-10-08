// Thin fetch wrapper over dashboard/backend/main.py's FastAPI endpoints.
// Every function here maps 1:1 to one backend route -- no client-side
// business logic, matching the backend's own "thin wrapper" design.

const BASE = "http://127.0.0.1:8000";

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  listSkus: () => request("/api/skus"),
  forecast: (sku, horizon = 7) => request(`/api/forecast?sku=${encodeURIComponent(sku)}&horizon=${horizon}`),
  sales: (sku, days = 28) => request(`/api/sales?sku=${encodeURIComponent(sku)}&days=${days}`),
  anomalies: (sku, lookbackDays = 90) =>
    request(`/api/anomalies?sku=${encodeURIComponent(sku)}&lookback_days=${lookbackDays}`),
  inventory: (sku) => request(`/api/inventory?sku=${encodeURIComponent(sku)}`),
  chat: (message, history, model) =>
    request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history, model }),
    }),
  commerceState: () => request("/api/commerce/state"),
  commerceLog: () => request("/api/commerce/log"),
  commerceOrders: () => request("/api/commerce/orders"),
  commerceDecisions: () => request("/api/commerce/decisions"),
  commerceAdvance: () => request("/api/commerce/advance", { method: "POST" }),
  commerceReset: () => request("/api/commerce/reset", { method: "POST" }),
};
