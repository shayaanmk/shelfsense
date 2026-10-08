import { useEffect, useState } from "react";
import { api } from "../api.js";
import ForecastChart from "./ForecastChart.jsx";
import AnomalyList from "./AnomalyList.jsx";

export default function SkuExplorer() {
  const [skus, setSkus] = useState([]);
  const [sku, setSku] = useState("");
  const [horizon, setHorizon] = useState(7);
  const [sales, setSales] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [anomalies, setAnomalies] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api
      .listSkus()
      .then((r) => {
        setSkus(r.skus);
        setSku(r.commerce_sku ?? r.skus[0]);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!sku) return;
    setLoading(true);
    setError(null);
    Promise.all([api.sales(sku, 42), api.forecast(sku, horizon), api.anomalies(sku, 90)])
      .then(([s, f, a]) => {
        setSales(s);
        setForecast(f);
        setAnomalies(a);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [sku, horizon]);

  return (
    <div className="panel">
      <div className="controls">
        <label>
          SKU:{" "}
          <select value={sku} onChange={(e) => setSku(e.target.value)}>
            {skus.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label>
          Forecast horizon:{" "}
          <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
            <option value={7}>7 days</option>
            <option value={28}>28 days</option>
          </select>
        </label>
      </div>

      {error && <p className="error">{error}</p>}
      {loading && <p className="muted">Loading...</p>}

      {!loading && sales && forecast && (
        <>
          <section className="card">
            <h3>Sales + forecast</h3>
            <ForecastChart sales={sales} forecast={forecast} />
            <p className="muted">
              Model: {forecast.model} &middot; as of {forecast.as_of_date} &middot; last{" "}
              {sales.summary.mean_daily_units.toFixed(1)} units/day avg
              {sales.summary.trend_pct_vs_prior_period != null &&
                ` (${sales.summary.trend_pct_vs_prior_period > 0 ? "+" : ""}${sales.summary.trend_pct_vs_prior_period}% vs prior period)`}
            </p>
          </section>

          <section className="card">
            <h3>Anomalies (last 90 days)</h3>
            <AnomalyList anomalies={anomalies} />
          </section>
        </>
      )}
    </div>
  );
}
